import AppKit
import WebKit

/// The Silent Whisper camera window: a floating panel that runs the web app's own Lip mode page (`lipmode.html?app`),
/// so the camera view, the lip tracker, its crop boxes and the read are exactly the browser's. The shortcut drives the
/// page's take (`window.baijiu.begin()` / `end()`); the page hands back the line it read through the "baijiu"
/// message handler. The panel never takes focus, so the user's text field keeps it.
@MainActor
final class LipWindow: NSObject, WKScriptMessageHandler, WKUIDelegate {
    enum Outcome { case line(String), failed(String) }

    private let server: URL
    private var panel: NSPanel?
    private var web: WKWebView?
    private var waiting: CheckedContinuation<Outcome, Never>?

    init(server: URL = LipReader.configuredServer()) {
        self.server = server
    }

    /// Opens the window (loading the page and its camera once) and brings it to the front.
    func show() {
        if panel == nil { build() }
        panel?.orderFrontRegardless()
    }

    /// Releases the camera (for Move Cursor, which needs it) by unloading the page.
    func close() {
        finish(.failed("The camera window closed."))
        web?.loadHTMLString("", baseURL: nil)
        panel?.orderOut(nil)
        panel = nil
        web = nil
    }

    /// Starts the page's take; false while the page or its camera is still starting.
    func begin() async -> Bool {
        show()
        guard let web else { return false }
        let started = try? await web.evaluateJavaScript("window.baijiu ? window.baijiu.begin() : false")
        return (started as? Bool) == true
    }

    /// Stops the take; returns the line the page read, or why it couldn't.
    func end() async -> Outcome {
        guard let web else { return .failed("The camera window is not open.") }
        finish(.failed("A new take replaced the previous one."))
        return await withCheckedContinuation { continuation in
            waiting = continuation
            web.evaluateJavaScript("window.baijiu ? (window.baijiu.end(), true) : false") { [weak self] value, _ in
                Task { @MainActor in
                    if (value as? Bool) != true { self?.finish(.failed("The camera page is still loading. Try again.")) }
                }
            }
            Task { @MainActor [weak self] in  // a read takes ~20 s; never wait forever
                try? await Task.sleep(for: .seconds(90))
                self?.finish(.failed("The read timed out."))
            }
        }
    }

    private func finish(_ outcome: Outcome) {
        waiting?.resume(returning: outcome)
        waiting = nil
    }

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
        guard let body = message.body as? [String: Any] else { return }
        if let text = body["text"] as? String, !text.isEmpty {
            finish(.line(text))
        } else {
            finish(.failed(body["error"] as? String ?? "No line read. Try again."))
        }
    }

    /// The page asks for the camera (getUserMedia); the app already holds the user's camera permission.
    func webView(_ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin,
                 initiatedByFrame frame: WKFrameInfo, type: WKMediaCaptureType,
                 decisionHandler: @escaping @MainActor (WKPermissionDecision) -> Void) {
        decisionHandler(.grant)
    }

    private func build() {
        let config = WKWebViewConfiguration()
        config.userContentController.add(self, name: "baijiu")
        config.mediaTypesRequiringUserActionForPlayback = []
        let web = WKWebView(frame: NSRect(x: 0, y: 0, width: 420, height: 470), configuration: config)
        web.uiDelegate = self
        var page = URLComponents(url: server.appendingPathComponent("lipmode.html"), resolvingAgainstBaseURL: false)!
        page.queryItems = [URLQueryItem(name: "app", value: nil), URLQueryItem(name: "user", value: "local")]
        web.load(URLRequest(url: page.url!))

        let panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 420, height: 470),
                            styleMask: [.titled, .closable, .utilityWindow, .nonactivatingPanel],
                            backing: .buffered, defer: false)
        panel.title = "Baijiu Machine · Lips"
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.hidesOnDeactivate = false
        panel.becomesKeyOnlyIfNeeded = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.contentView = web
        if let visible = NSScreen.main?.visibleFrame {
            panel.setFrameOrigin(NSPoint(x: visible.maxX - 440, y: visible.maxY - 490))
        }
        self.panel = panel
        self.web = web
    }
}
