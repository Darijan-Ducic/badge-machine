import Foundation

struct Gemini {
    let key: String
    private let model = "gemini-3.8-flash"

    func transcribe(_ clip: URL) async throws -> String? {
        let video = try Data(contentsOf: clip)
        let body: [String: Any] = [
            "contents": [["parts": [
                ["inlineData": ["mimeType": "video/mov", "data": video.base64EncodedString()],
                 "videoMetadata": ["fps": 12]],
                ["text": """
                 Read the lip movements in this silent video. Return only the words visibly articulated,                  in their original language. If there is no clear speech or you cannot read it,                  return exactly UNREADABLE. Do not add context or guess missing words.
                 """]
            ]]]
        ]
        let url = URL(string: "https://generativelanguage.googleapis.com/v1beta/models/\(model):generateContent")!
        var request = URLRequest(url: url, timeoutInterval: 40)
        request.httpMethod = "POST"
        request.setValue(key, forHTTPHeaderField: "x-goog-api-key")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw ResponseError.rejected((response as? HTTPURLResponse)?.statusCode ?? 0)
        }
        let result = try JSONDecoder().decode(Response.self, from: data)
        let text = result.candidates?.first?.content?.parts
            .filter { $0.thought != true }.compactMap(\.text).joined()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard let text, !text.isEmpty, text != "UNREADABLE" else { return nil }
        return text
    }

    private struct Response: Decodable {
        struct Candidate: Decodable {
            struct Content: Decodable {
                struct Part: Decodable { let text: String?; let thought: Bool? }
                let parts: [Part]
            }
            let content: Content?
        }
        let candidates: [Candidate]?
    }

    private enum ResponseError: LocalizedError {
        case rejected(Int)
        var errorDescription: String? {
            switch self {
            case .rejected(let code): "Gemini ha rifiutato la richiesta (HTTP \(code))."
            }
        }
    }
}
