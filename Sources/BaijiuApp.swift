import AppKit
import SwiftUI

@main
struct BaijiuApp: App {
    @StateObject private var studio = Studio()
    @Environment(\.openWindow) private var openWindow

    private let menuMark: NSImage = {
        let image = Bundle.main.image(forResource: "MenuMark") ?? NSImage()
        image.size = NSSize(width: 17, height: 17)
        image.isTemplate = true
        return image
    }()

    var body: some Scene {
        Window("Baijiu Machine", id: "studio") {
            StudioView(studio: studio)
                .preferredColorScheme(.light)
                .task { await studio.requestCamera() }
        }
        .windowResizability(.contentSize)

        MenuBarExtra {
            if !studio.status.isEmpty { Text(studio.status) }
            Button("Open Baijiu Machine") {
                NSApp.activate(ignoringOtherApps: true)
                openWindow(id: "studio")
            }
            Button("Quit") { NSApp.terminate(nil) }
        } label: {
            Image(nsImage: menuMark)
                .opacity(studio.phase == .recording ? 0.55 : 1)
                .accessibilityLabel(studio.phase == .recording ? "Baijiu Machine, recording" : "Baijiu Machine")
        }
    }
}

struct StudioView: View {
    @ObservedObject var studio: Studio

    private let ink = Color(white: 0.08)
    private let paper = Color(red: 0.965, green: 0.965, blue: 0.953)

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if studio.choosing {
                VStack(spacing: 14) {
                    Text(studio.pendingShortcut?.label ?? "Press a key…")
                        .font(.system(size: 24, weight: .medium, design: .monospaced))
                        .accessibilityLabel(studio.pendingShortcut.map { "Selected key: \($0.label)" } ?? "Press a key")
                        .frame(maxWidth: .infinity, minHeight: 48)
                        .background(.white, in: RoundedRectangle(cornerRadius: 8))
                        .background {
                            ShortcutKeyCapture(onKeyDown: studio.recordShortcut)
                                .accessibilityHidden(true)
                        }

                    if studio.pendingShortcut != nil {
                        Button("Confirm", action: studio.confirmShortcut)
                            .buttonStyle(.borderedProminent)
                            .tint(ink)
                    }
                }
            } else {
                HStack(spacing: 10) {
                    Text("Keyboard shortcut").font(.system(size: 14, weight: .medium))
                    Spacer()
                    Button(studio.shortcut?.label ?? "Choose Shortcut", action: studio.chooseShortcut)
                        .buttonStyle(.borderedProminent)
                        .tint(ink)
                        .accessibilityLabel(studio.shortcut.map { "Change shortcut, currently \($0.label)" } ?? "Choose Shortcut")
                        .disabled(studio.phase != .ready)
                }
            }

            if !studio.status.isEmpty {
                Text(studio.status)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(22)
        .frame(width: 370)
        .foregroundStyle(ink)
        .background(paper)
        .onDisappear {
            if studio.choosing { studio.cancelChoosing() }
        }
    }
}

private struct ShortcutKeyCapture: NSViewRepresentable {
    let onKeyDown: (NSEvent) -> Void

    func makeNSView(context: Context) -> KeyView {
        let view = KeyView()
        view.onKeyDown = onKeyDown
        return view
    }

    func updateNSView(_ view: KeyView, context: Context) {
        view.onKeyDown = onKeyDown
    }

    final class KeyView: NSView {
        var onKeyDown: ((NSEvent) -> Void)?
        override var acceptsFirstResponder: Bool { true }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            guard let window else { return }
            NSApp.activate(ignoringOtherApps: true)
            window.makeKeyAndOrderFront(nil)
            window.makeFirstResponder(self)
        }

        override func keyDown(with event: NSEvent) {
            onKeyDown?(event)
        }

        override func performKeyEquivalent(with event: NSEvent) -> Bool {
            guard window?.firstResponder === self else { return false }
            onKeyDown?(event)
            return true
        }
    }
}
