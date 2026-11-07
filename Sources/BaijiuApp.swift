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
                .onReceive(NotificationCenter.default.publisher(for: NSApplication.willTerminateNotification)) { _ in
                    studio.discardClip()
                }
        }
        .windowResizability(.contentSize)

        MenuBarExtra {
            Text(studio.status)
            Button(studio.action) { studio.toggle() }
                .disabled(studio.busy)
            Button("Apri Baijiu Machine") {
                openWindow(id: "studio")
                NSApp.activate(ignoringOtherApps: true)
            }
            Divider()
            Button("Esci") {
                studio.discardClip()
                NSApp.terminate(nil)
            }
        } label: {
            Image(systemName: studio.active ? "record.circle" : "viewfinder")
                .accessibilityLabel("Baijiu Machine")
        }
    }
}

struct StudioView: View {
    @ObservedObject var studio: Studio
    private let ink = Color(white: 0.08)
    private let paper = Color(red: 0.965, green: 0.965, blue: 0.953)

    var body: some View {
        VStack(alignment: .leading, spacing: 23) {
            HStack {
                if let icon = Bundle.main.image(forResource: "Mark") {
                    Image(nsImage: icon).resizable().frame(width: 46, height: 46)
                        .accessibilityHidden(true)
                }
                stamp("BAIJIU MACHINE")
                Spacer()
                stamp("MACOS / 01")
                    .foregroundStyle(.secondary)
            }

            Rectangle().fill(ink.opacity(0.16)).frame(height: 1)

            VStack(alignment: .leading, spacing: 9) {
                Text("Parla in silenzio.")
                    .font(.system(size: 57, weight: .regular, design: .serif))
                    .tracking(-3)
                    .accessibilityAddTraits(.isHeader)
                Text("Guarda la fotocamera e articola una frase breve.")
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 10) {
                Button(action: studio.toggle) {
                    HStack(spacing: 10) {
                        Image(systemName: studio.phase == .recording ? "stop.fill" : "record.circle")
                        Text(studio.action)
                    }
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 18)
                    .frame(height: 41)
                    .background(ink, in: Capsule())
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.return, modifiers: [])
                .disabled(studio.busy)
                .opacity(studio.busy ? 0.45 : 1)

                if studio.busy {
                    ProgressView().controlSize(.small)
                }
                stamp(studio.status.uppercased())
                    .foregroundStyle(.secondary)
                Spacer()
                if studio.phase == .recording {
                    TimelineView(.periodic(from: studio.startedAt, by: 1)) { context in
                        let elapsed = max(0, min(8, Int(context.date.timeIntervalSince(studio.startedAt))))
                        stamp(String(format: "00:%02d / 00:08", elapsed))
                    }
                }
            }

            VStack(alignment: .leading, spacing: 13) {
                HStack {
                    stamp("TRASCRIZIONE")
                    Spacer()
                    if !studio.transcript.isEmpty { Button("Copia", action: studio.copy) }
                    if studio.clip != nil && !studio.busy { Button("Salva clip…", action: studio.saveClip) }
                }
                ScrollView {
                    Text(result)
                        .font(.system(size: studio.transcript.isEmpty ? 15 : 25, design: .serif))
                        .foregroundStyle(studio.transcript.isEmpty ? .secondary : .primary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding(17)
            .frame(height: 135)
            .background(.white)
            .overlay(Rectangle().strokeBorder(ink.opacity(0.15)))

            HStack {
                stamp("SOLO VIDEO")
                Spacer()
                stamp(studio.online ? "TRASCRIZIONE VIA GEMINI" : "REGISTRAZIONE LOCALE")
            }
            .foregroundStyle(.secondary)
        }
        .padding(30)
        .frame(width: 560)
        .foregroundStyle(ink)
        .background(paper)
    }

    private var result: String {
        switch studio.phase {
        case .done: studio.transcript
        case .failed: studio.error
        case .unclear: "Prova di nuovo con il viso ben illuminato."
        case .transcribing: "Sto leggendo il labiale…"
        case .offline: "Clip senza audio pronta. Configura Gemini per ottenere il testo."
        default: "Il testo comparirà qui."
        }
    }

    private func stamp(_ text: String) -> some View {
        Text(text).font(.system(size: 10, weight: .medium, design: .monospaced)).tracking(1)
    }
}
