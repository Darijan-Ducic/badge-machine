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
                openWindow(id: "studio")
                NSApp.activate(ignoringOtherApps: true)
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
    @State private var keyMonitor: Any?

    private let ink = Color(white: 0.08)
    private let paper = Color(red: 0.965, green: 0.965, blue: 0.953)

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 10) {
                Text("Keyboard shortcut").font(.system(size: 14, weight: .medium))
                Spacer()

                if studio.choosing {
                    Text(studio.pendingShortcut?.label ?? "Press a key…")
                        .font(.system(size: 13, design: .monospaced))
                    Button("Confirm", action: studio.confirmShortcut)
                        .buttonStyle(.borderedProminent)
                        .tint(ink)
                        .disabled(studio.pendingShortcut == nil)
                } else {
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
        .onAppear {
            keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
                guard studio.choosing else { return event }
                let shortcutModifiers: NSEvent.ModifierFlags = [.command, .control, .option, .shift]
                if event.modifierFlags.intersection(shortcutModifiers).isEmpty {
                    if event.keyCode == 48 { return event }
                    if event.keyCode == 36, studio.pendingShortcut != nil {
                        if !event.isARepeat { studio.confirmShortcut() }
                        return nil
                    }
                }
                studio.recordShortcut(event)
                return nil
            }
        }
        .onDisappear {
            if let keyMonitor { NSEvent.removeMonitor(keyMonitor) }
            keyMonitor = nil
            if studio.choosing { studio.cancelChoosing() }
        }
    }
}
