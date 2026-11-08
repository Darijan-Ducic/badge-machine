import Foundation

// End-to-end check against the running server (costs one real read). From the project root:
// swiftc -parse-as-library Sources/LipBoxes.swift Sources/LipReader.swift Tests/LipReaderLive.swift -o /tmp/baijiu-lip-reader-live
// /tmp/baijiu-lip-reader-live path/to/recording.mov
@main
struct LipReaderLive {
    static func main() async throws {
        setvbuf(stdout, nil, _IOLBF, 0)
        guard CommandLine.arguments.count == 2 else {
            print("usage: baijiu-lip-reader-live <recording.mov>")
            exit(2)
        }
        let movie = URL(fileURLWithPath: CommandLine.arguments[1])
        let started = Date()
        let boxes = try await LipBoxes.scan(movie)
        print("lip boxes: \(boxes.count) frames in \(Int(Date().timeIntervalSince(started) * 1000)) ms")
        print("first: \(boxes.first ?? []) last: \(boxes.last ?? [])")

        var reader = LipReader()
        let send = reader.send
        reader.send = { request in
            let (data, code) = try await send(request)
            print("POST \(request.url!.absoluteString) -> HTTP \(code)")
            print(String(decoding: data, as: UTF8.self))
            return (data, code)
        }
        let read = try await reader.read(movie, boxes: boxes)
        print("decoded from \(read.endpoint), clip \(read.clip)")
        for (index, lines) in read.segments.enumerated() {
            print("segment \(index): " + lines.map { "\($0.text) (\(String(format: "%.3f", $0.p)))" }.joined(separator: " | "))
        }
        print("text: \(read.text)")
    }
}
