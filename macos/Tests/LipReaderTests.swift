import Foundation

// Run from the project root:
// swiftc -parse-as-library Sources/LipBoxes.swift Sources/LipReader.swift Tests/LipReaderTests.swift -o /tmp/baijiu-lip-reader-tests
// /tmp/baijiu-lip-reader-tests
@main
struct LipReaderTests {
    nonisolated(unsafe) static var failures = 0

    static func check(_ name: String, _ passed: Bool, _ detail: @autoclosure () -> String = "") {
        print(passed ? "PASS \(name)" : "FAIL \(name) \(detail())")
        if !passed { failures += 1 }
    }

    static func near(_ a: [Double]?, _ b: [Double]) -> Bool {
        guard let a, a.count == b.count else { return false }
        return zip(a, b).allSatisfy { abs($0 - $1) < 1e-6 }
    }

    static func main() async {
        // Vision's lower-left normalized points -> top-left recording pixels, padded 1.6x wide, 0.52 tall/wide.
        // Lips span x 0.4...0.6 (256...384 px) and y 0.30...0.35 up from the bottom (312...336 px down from the top).
        let closed = LipBoxes.crop(lips: [CGPoint(x: 0.4, y: 0.32), CGPoint(x: 0.6, y: 0.33),
                                          CGPoint(x: 0.5, y: 0.35), CGPoint(x: 0.5, y: 0.30)],
                                   in: CGSize(width: 640, height: 480))
        check("closed lips: centered on the lips, y flipped to a top-left origin, padded to the tracker's shape",
              near(closed, [320 - 102.4, 324 - 53.248, 204.8, 106.496]), "\(String(describing: closed))")

        // An open mouth (80 px tall; 1.6 x 80 = 128 > 0.52 x 204.8) gets height = 1.6 x lips height, still centered.
        let open = LipBoxes.crop(lips: [CGPoint(x: 0.4, y: 0.30), CGPoint(x: 0.6, y: 0.30),
                                        CGPoint(x: 0.5, y: 0.30 + 80.0 / 480)],
                                 in: CGSize(width: 640, height: 480))
        check("open mouth: the crop grows to 1.6 x the lips' height",
              near(open, [217.6, 232, 204.8, 128]), "\(String(describing: open))")

        check("no measurable mouth width gives no box",
              LipBoxes.crop(lips: [CGPoint(x: 0.5, y: 0.3), CGPoint(x: 0.5, y: 0.4)], in: CGSize(width: 640, height: 480)) == nil)

        let points = LipBoxes.imagePoints([CGPoint(x: 0.5, y: 0.25)], face: CGRect(x: 0.25, y: 0.2, width: 0.5, height: 0.6))
        check("face-relative landmark -> image-normalized point",
              abs(points[0].x - 0.5) < 1e-9 && abs(points[0].y - 0.35) < 1e-9, "\(points)")

        // POST /vtt/paragraph (contract): the paragraph is each segment's top line, as sentences.
        let paragraph = Data(#"""
        {"clip": "1794104081162842001",
         "segments": [{"lines": [{"text": "hey guys I'm Darian", "p": 0.61}, {"text": "this is intense", "p": 0.2},
                                 {"text": "I love today's hackathon", "p": 0.1}], "frames": [0, 52]},
                      {"lines": [{"text": "I'm a founder of Draki", "p": 0.4}, {"text": "our team is three people", "p": 0.3},
                                 {"text": "coffee is keeping us alive", "p": 0.1}], "frames": [60, 120]}],
         "paragraph": ["hey guys I'm Darian", "I'm a founder of Draki"], "latency_ms": 31000}
        """#.utf8)
        do {
            let read = try LipReader.decodeParagraph(paragraph)
            check("paragraph: clip, top-3 per segment, the paragraph in order",
                  read.clip == "1794104081162842001" && read.segments.map(\.count) == [3, 3]
                    && read.segments[1][1] == LipReader.Line(text: "our team is three people", p: 0.3)
                    && read.paragraph == ["hey guys I'm Darian", "I'm a founder of Draki"], "\(read)")
            check("paragraph text: capitalized sentences, no words added",
                  read.text == "Hey guys I'm Darian. I'm a founder of Draki.", read.text)
        } catch {
            check("paragraph decodes", false, "\(error)")
        }

        // POST /vtt (a real response from the running server): the paragraph is lines[0].
        let vtt = Data(#"""
        {"clip": "1794104081162842000", "vtt": {"text": "I'm a founder of Draki", "p": 0.30692997820449514}, "action": "confirm",
         "alternatives": [{"text": "I'm a founder of Draki", "p": 0.30692997820449514}, {"text": "how are you today", "p": 0.05310897427102449},
                          {"text": "hey guys I'm Darian", "p": 0.05041837198986465}],
         "lines": [{"text": "I'm a founder of Draki", "p": 0.30692997820449514}, {"text": "hey guys I'm Darian", "p": 0.05041837198986465},
                   {"text": "I want to bring your real life context to your AI", "p": 0.034711635382150524}],
         "gemini_lines": [{"text": "I'm a founder of Draki", "p": 0.7689754290233427}], "latency_ms": 24973}
        """#.utf8)
        do {
            let read = try LipReader.decodeVTT(vtt)
            check("vtt: one segment of the top-3 phrase-list lines, paragraph = lines[0]",
                  read.clip == "1794104081162842000" && read.segments.count == 1 && read.segments[0].count == 3
                    && read.paragraph == ["I'm a founder of Draki"] && read.text == "I'm a founder of Draki.", "\(read)")
            let empty = try LipReader.decodeVTT(Data(#"{"clip": "1", "vtt": null, "lines": [], "alternatives": []}"#.utf8))
            check("vtt with no lines reads nothing", empty.paragraph.isEmpty && empty.text.isEmpty, "\(empty)")
        } catch {
            check("vtt decodes", false, "\(error)")
        }

        // Endpoint choice: /vtt/paragraph first; /vtt only when the server has no paragraph route (404).
        let movie = FileManager.default.temporaryDirectory.appendingPathComponent("lip-reader-test.mov")
        try? Data("movie".utf8).write(to: movie)
        let boxes = [[0, 328, 252, 83, 43], [33, 327, 251, 83, 43]]
        func reader(_ answers: [String: (Int, Data)], log: @escaping (URLRequest) -> Void) -> LipReader {
            var reader = LipReader(server: URL(string: "http://127.0.0.1:8765")!, user: "local")
            reader.send = { request in
                log(request)
                let (code, body) = answers[request.url!.path] ?? (500, Data())
                return (body, code)
            }
            return reader
        }

        final class Log: @unchecked Sendable { var requests: [URLRequest] = [] }
        let fallbackLog = Log()
        do {
            let read = try await reader(["/vtt/paragraph": (404, Data("Not Found".utf8)), "/vtt": (200, vtt)]) {
                fallbackLog.requests.append($0)
            }.read(movie, boxes: boxes)
            let urls = fallbackLog.requests.map { $0.url!.absoluteString }
            let headers = fallbackLog.requests.last.map {
                ($0.value(forHTTPHeaderField: "X-Tracker-Boxes"), $0.value(forHTTPHeaderField: "X-Tracker-Flags"),
                 $0.httpBody)
            }
            check("404 on /vtt/paragraph falls back to /vtt with the same body and headers",
                  urls == ["http://127.0.0.1:8765/vtt/paragraph?user=local", "http://127.0.0.1:8765/vtt?user=local"]
                    && read.endpoint == "/vtt"
                    && headers?.0 == "[[0,328,252,83,43],[33,327,251,83,43]]" && headers?.1 == #"["tracker:vision"]"#
                    && headers?.2 == Data("movie".utf8), "\(urls) \(String(describing: headers))")
        } catch {
            check("404 fallback", false, "\(error)")
        }

        let failureLog = Log()
        do {
            _ = try await reader(["/vtt/paragraph": (500, Data(#"{"error": "ffmpeg failed"}"#.utf8)), "/vtt": (200, vtt)]) {
                failureLog.requests.append($0)
            }.read(movie, boxes: boxes)
            check("a paragraph-endpoint failure is reported, not hidden behind /vtt", false, "no error thrown")
        } catch {
            check("a paragraph-endpoint failure is reported, not hidden behind /vtt",
                  failureLog.requests.count == 1 && error.localizedDescription.contains("ffmpeg failed"),
                  "\(failureLog.requests.count) requests, \(error.localizedDescription)")
        }

        let paragraphLog = Log()
        do {
            let read = try await reader(["/vtt/paragraph": (200, paragraph)]) { paragraphLog.requests.append($0) }
                .read(movie, boxes: boxes)
            check("a server with /vtt/paragraph is read once there", paragraphLog.requests.count == 1
                  && read.endpoint == "/vtt/paragraph" && read.paragraph.count == 2, "\(read)")
        } catch {
            check("paragraph endpoint", false, "\(error)")
        }

        try? FileManager.default.removeItem(at: movie)
        print(failures == 0 ? "All lip reader checks passed." : "\(failures) lip reader check(s) failed.")
        exit(failures == 0 ? 0 : 1)
    }
}
