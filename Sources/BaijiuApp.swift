import AppKit
import SwiftUI

@main
struct BaijiuApp: App {
    @StateObject private var studio = Studio()
    @Environment(\.openWindow) private var openWindow

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
            Image(systemName: studio.phase == .recording ? "record.circle.fill" : "viewfinder")
                .accessibilityLabel("Baijiu Machine")
        }
    }
}

struct StudioView: View {
    @ObservedObject var studio: Studio
    @State private var keyMonitor: Any?

    private let ink = Color(white: 0.08)
    private let paper = Color(red: 0.965, green: 0.965, blue: 0.953)

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            HStack(spacing: 12) {
                if let icon = Bundle.main.image(forResource: "Mark") {
                    Image(nsImage: icon).resizable().frame(width: 46, height: 46)
                        .accessibilityHidden(true)
                }
                Text("BAIJIU MACHINE")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .tracking(1)
            }

            Rectangle().fill(ink.opacity(0.15)).frame(height: 1)

            VStack(alignment: .leading, spacing: 9) {
                Text("Speak silently.")
                    .font(.system(size: 54, design: .serif))
                    .tracking(-2.5)
                    .accessibilityAddTraits(.isHeader)
                Text("Press your shortcut to start recording. Press it again to type where your cursor is.")
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack {
                Text("Keyboard shortcut").font(.system(size: 14, weight: .medium))
                Spacer()
                Button(studio.choosing ? "Press keys…" : studio.shortcut?.label ?? "Choose Shortcut") {
                    studio.chooseShortcut()
                }
                .buttonStyle(.borderedProminent)
                .tint(ink)
                .accessibilityLabel(studio.shortcut.map { "Change shortcut, currently \($0.label)" } ?? "Choose Shortcut")
                .disabled(studio.choosing || studio.phase != .ready)
            }

            if !studio.status.isEmpty {
                Text(studio.status)
                    .font(.system(size: 12))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(30)
        .frame(width: 460)
        .foregroundStyle(ink)
        .background(paper)
        .onAppear {
            keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
                guard studio.choosing else { return event }
                studio.recordShortcut(event)
                return nil
            }
        }
        .onDisappear {
            if let keyMonitor { NSEvent.removeMonitor(keyMonitor) }
            keyMonitor = nil
            studio.cancelChoosing()
        }
    }
}
