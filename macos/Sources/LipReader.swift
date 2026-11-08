import Foundation

/// The badge-machine server's lip mode: one muted recording plus its mouth boxes in, the lines it read out.
/// `POST /vtt/paragraph` reads one line per segment; while a server lacks it (404) `POST /vtt` reads one line.
struct LipReader {
    static let defaultServer = URL(string: "http://127.0.0.1:8765")!

    /// One line choice the server ranked: its text from the user's phrase list and its probability.
    struct Line: Decodable, Equatable {
        let text: String
        let p: Double
    }

    /// A read: per segment its top-ranked lines (top 3), and the top line of each segment in order.
    struct Read: Equatable {
        let clip: String
        let segments: [[Line]]
        let paragraph: [String]
        let endpoint: String

        /// The paragraph as sentences: each read line capitalized and closed with a period, nothing else added.
        var text: String { LipReader.sentences(paragraph) }
    }

    typealias Send = (URLRequest) async throws -> (Data, Int)

    let server: URL
    let user: String
    var send: Send = LipReader.urlSession

    init(server: URL = LipReader.configuredServer(), user: String = "local") {
        self.server = server
        self.user = user
    }

    /// `BAIJIU_SERVER_URL` in the environment, else the `serverURL` user default, else the local server.
    static func configuredServer(environment: [String: String] = ProcessInfo.processInfo.environment,
                                 defaults: UserDefaults = .standard) -> URL {
        for raw in [environment["BAIJIU_SERVER_URL"], defaults.string(forKey: "serverURL")] {
            if let raw, let url = URL(string: raw.trimmingCharacters(in: .whitespacesAndNewlines)),
               url.scheme != nil, url.host != nil { return url }
        }
        return defaultServer
    }

    func read(_ movie: URL, boxes: [[Int]]) async throws -> Read {
        let video = try Data(contentsOf: movie)
        let (data, code) = try await send(request("vtt/paragraph", video: video, boxes: boxes))
        if code != 404 {
            try Self.check(data, code)
            return try Self.decodeParagraph(data)
        }
        let (fallback, fallbackCode) = try await send(request("vtt", video: video, boxes: boxes))
        try Self.check(fallback, fallbackCode)
        return try Self.decodeVTT(fallback)
    }

    func request(_ path: String, video: Data, boxes: [[Int]]) throws -> URLRequest {
        var components = URLComponents(url: server.appendingPathComponent(path), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "user", value: user)]
        var request = URLRequest(url: components.url!, timeoutInterval: 180)
        request.httpMethod = "POST"
        request.setValue("video/quicktime", forHTTPHeaderField: "Content-Type")
        request.setValue(String(decoding: try JSONEncoder().encode(boxes), as: UTF8.self),
                         forHTTPHeaderField: "X-Tracker-Boxes")
        request.setValue(#"["tracker:vision"]"#, forHTTPHeaderField: "X-Tracker-Flags")
        request.httpBody = video
        return request
    }

    static func decodeParagraph(_ data: Data) throws -> Read {
        struct Segment: Decodable { let lines: [Line] }
        struct Body: Decodable {
            let clip: Clip
            let segments: [Segment]
            let paragraph: [String]?
        }
        let body = try JSONDecoder().decode(Body.self, from: data)
        let segments = body.segments.map(\.lines)
        return Read(clip: body.clip.id, segments: segments,
                    paragraph: body.paragraph ?? segments.compactMap { $0.first?.text }, endpoint: "/vtt/paragraph")
    }

    static func decodeVTT(_ data: Data) throws -> Read {
        struct Body: Decodable {
            let clip: Clip
            let lines: [Line]
        }
        let body = try JSONDecoder().decode(Body.self, from: data)
        return Read(clip: body.clip.id, segments: body.lines.isEmpty ? [] : [body.lines],
                    paragraph: body.lines.first.map { [$0.text] } ?? [], endpoint: "/vtt")
    }

    static func sentences(_ lines: [String]) -> String {
        lines.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }.map { line in
            let capitalized = line.prefix(1).uppercased() + line.dropFirst()
            return ".!?".contains(capitalized.last!) ? capitalized : capitalized + "."
        }.joined(separator: " ")
    }

    /// The server's clip id: a string, or a number from older servers.
    private struct Clip: Decodable {
        let id: String
        init(from decoder: Decoder) throws {
            let value = try decoder.singleValueContainer()
            id = try (try? value.decode(String.self)) ?? String(value.decode(Int.self))
        }
    }

    private static func check(_ data: Data, _ code: Int) throws {
        guard !(200..<300).contains(code) else { return }
        let message = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["error"] as? String
        throw ReadError.rejected(code, message)
    }

    private static func urlSession(_ request: URLRequest) async throws -> (Data, Int) {
        let (data, response) = try await URLSession.shared.data(for: request)
        return (data, (response as? HTTPURLResponse)?.statusCode ?? 0)
    }

    enum ReadError: LocalizedError {
        case rejected(Int, String?)
        var errorDescription: String? {
            switch self {
            case .rejected(let code, let message):
                "The lip reader failed (HTTP \(code))" + (message.map { ": \($0)" } ?? ".")
            }
        }
    }
}
