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
            Button(studio.phase == .handCursor ? "Stop Moving Cursor" : "Start Moving Cursor",
                   action: studio.toggleHandCursor)
                .disabled(!studio.canToggleHandCursor)
            Button("Open Baijiu Machine") {
                NSApp.activate(ignoringOtherApps: true)
                openWindow(id: "studio")
                studio.resumeShortcutMonitoring()
            }
            Button("Quit") { NSApp.terminate(nil) }
        } label: {
            Image(nsImage: menuMark)
                .opacity(studio.phase == .recording || studio.phase == .handCursor ? 0.55 : 1)
                .accessibilityLabel(studio.phase == .handCursor ? "Baijiu Machine, moving pointer" : studio.phase == .recording ? "Baijiu Machine, recording" : "Baijiu Machine")
        }
    }
}

struct StudioView: View {
    @ObservedObject var studio: Studio

    private let ink = Color(white: 0.08)
    private let paper = Color(red: 0.965, green: 0.965, blue: 0.953)

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let mode = studio.editingMode {
                Text(mode.rawValue)
                    .font(.system(size: 14, weight: .semibold))
                VStack(spacing: 14) {
                    Text(studio.pendingShortcut?.label ?? "Press a key…")
                        .font(.system(size: 24, weight: .medium, design: .monospaced))
                        .accessibilityLabel(studio.pendingShortcut.map { "Selected key: \($0.label)" } ?? "Press a key")
                        .frame(maxWidth: .infinity, minHeight: 48)
                        .background(.white, in: RoundedRectangle(cornerRadius: 8))
                        .background {
                            ShortcutKeyCapture(onKeyDown: studio.recordShortcut, onModifier: studio.previewModifier)
                                .accessibilityHidden(true)
                        }

                    if studio.fnConfirmationNeedsPermission {
                        Text("Confirm to request Input Monitoring for Fn.")
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                    }

                    if studio.pendingShortcut != nil {
                        Button("Confirm", action: studio.confirmShortcut)
                            .buttonStyle(.borderedProminent)
                            .tint(ink)
                    }
                }
            } else {
                ForEach(Studio.Mode.allCases, id: \.self) { mode in
                    if mode == .cursor { Divider() }
                    HStack(alignment: .center, spacing: 16) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(mode.rawValue).font(.system(size: 14, weight: .semibold))
                            Text(mode.description)
                                .font(.system(size: 12))
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        Spacer(minLength: 8)
                        let shortcut = studio.shortcuts[mode]
                        Button(shortcut?.label ?? "Choose Shortcut") {
                            studio.chooseShortcut(for: mode)
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(ink)
                        .accessibilityLabel(shortcut.map { "Change \(mode.rawValue) shortcut, currently \($0.label)" }
                            ?? "Choose Shortcut for \(mode.rawValue)")
                        .disabled(studio.phase != .ready)
                    }
                }
            }

            if studio.needsFnSetup {
                Divider()
                Text("Finish setting up Fn")
                    .font(.system(size: 13, weight: .semibold))
                Text("macOS requires Input Monitoring so Baijiu Machine can detect Fn while you use another app.")
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Button("Open Input Monitoring", action: studio.openInputMonitoring)
            }

            if studio.needsModifierSetup {
                Button("Open Accessibility Settings", action: studio.openModifierAccessibility)
            }

            if !studio.status.isEmpty {
                Text(studio.status)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(22)
        .frame(width: 435)
        .foregroundStyle(ink)
        .background(paper)
        .onAppear { studio.resumeShortcutMonitoring() }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            studio.resumeShortcutMonitoring()
        }
        .onDisappear {
            if studio.choosing { studio.cancelChoosing() }
        }
    }
}

private struct ShortcutKeyCapture: NSViewRepresentable {
    let onKeyDown: (NSEvent) -> Void
    let onModifier: (Shortcut) -> Void

    func makeNSView(context: Context) -> KeyView {
        let view = KeyView()
        view.onKeyDown = onKeyDown
        view.onModifier = onModifier
        return view
    }

    func updateNSView(_ view: KeyView, context: Context) {
        view.onKeyDown = onKeyDown
        view.onModifier = onModifier
    }

    final class KeyView: NSView {
        var onKeyDown: ((NSEvent) -> Void)?
        var onModifier: ((Shortcut) -> Void)?
        private var monitor: Any?
        override var acceptsFirstResponder: Bool { true }

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            if let monitor { NSEvent.removeMonitor(monitor) }
            monitor = nil
            guard let window else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .flagsChanged]) { [weak self] event in
                guard let self, self.window?.isKeyWindow == true else { return event }
                if event.type == .flagsChanged {
                    if let modifier = Shortcut.pressedModifier(event) {
                        self.onModifier?(modifier)
                    }
                } else {
                    self.onKeyDown?(event)
                }
                return nil
            }
            NSApp.activate(ignoringOtherApps: true)
            window.makeKeyAndOrderFront(nil)
            window.makeFirstResponder(self)
        }

        deinit {
            if let monitor { NSEvent.removeMonitor(monitor) }
        }
    }
}
