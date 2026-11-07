import Foundation

struct APIKeys {
    let gemini: String?
    let meta: String?

    init(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        bundleURL: URL = Bundle.main.bundleURL
    ) {
        let buildDirectory = bundleURL.deletingLastPathComponent()
        let local = buildDirectory.lastPathComponent == "build"
            ? Self.readLocalKeys(in: buildDirectory.deletingLastPathComponent())
            : [:]
        gemini = Self.key(from: environment["GEMINI_API_KEY"]) ?? local["GEMINI_API_KEY"]
        meta = Self.key(from: environment["MODEL_API_KEY"]) ?? local["MODEL_API_KEY"]
    }

    private static func readLocalKeys(in directory: URL) -> [String: String] {
        let file = directory.appendingPathComponent(".env.local")
        guard let contents = try? String(contentsOf: file, encoding: .utf8) else { return [:] }
        var keys: [String: String] = [:]
        for line in contents.split(whereSeparator: \.isNewline) {
            let parts = line.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2 else { continue }
            let name = parts[0].trimmingCharacters(in: .whitespaces)
            guard name == "GEMINI_API_KEY" || name == "MODEL_API_KEY" else { continue }
            keys[name] = key(from: String(parts[1]))
        }
        return keys
    }

    private static func key(from raw: String?) -> String? {
        guard let value = raw?.trimmingCharacters(in: .whitespacesAndNewlines),
              !value.isEmpty else { return nil }
        if value.count >= 2, let first = value.first, first == value.last,
           first == "\"" || first == "'" {
            return value.count == 2 ? nil : String(value.dropFirst().dropLast())
        }
        return value
    }
}
