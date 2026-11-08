import AppKit
import SwiftUI

/// A small floating badge next to the pointer while Silent Whisper records and reads, so the user can see the app is
/// working while their cursor sits in another app's text field. It never takes focus, so the paste lands in that field.
@MainActor
final class ProcessingHUD {
    enum State: Equatable { case recording, reading, done(String), copied(String), failed(String) }

    private var panel: NSPanel?
    private var hideTask: Task<Void, Never>?

    func show(_ state: State) {
        hideTask?.cancel()
        let panel = self.panel ?? Self.makePanel()
        self.panel = panel
        panel.ignoresMouseEvents = true
        let view = NSHostingView(rootView: HUDView(state: state))
        panel.contentView = view
        panel.setContentSize(view.fittingSize)
        Self.place(panel)
        panel.orderFrontRegardless()
        switch state {
        case .done, .copied, .failed:
            hideTask = Task { [weak self] in
                try? await Task.sleep(for: .seconds(2.5))
                guard !Task.isCancelled else { return }
                self?.hide()
            }
        case .recording, .reading:
            break
        }
    }

    /// Lip mode's card: "You said one of these" with the read's top lines, the pasted one marked. A click calls
    /// `onPick` (the panel never takes focus, so the user's text field keeps it). Stays up for 12 s.
    func showOptions(_ lines: [String], pasted: String?, note: String, onPick: @escaping (String) -> Void) {
        hideTask?.cancel()
        let panel = self.panel ?? Self.makePanel()
        self.panel = panel
        panel.ignoresMouseEvents = false
        let view = FirstClickHostingView(rootView: OptionsView(lines: lines, pasted: pasted, note: note) { [weak self] pick in
            onPick(pick)
            self?.show(.done(pick))
        })
        panel.contentView = view
        panel.setContentSize(view.fittingSize)
        Self.place(panel)
        panel.orderFrontRegardless()
        hideTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(12))
            guard !Task.isCancelled else { return }
            self?.hide()
        }
    }

    func hide() {
        hideTask?.cancel()
        panel?.orderOut(nil)
    }

    private static func makePanel() -> NSPanel {
        let panel = NSPanel(contentRect: .zero, styleMask: [.nonactivatingPanel, .borderless], backing: .buffered,
                            defer: true)
        panel.isFloatingPanel = true
        panel.level = .statusBar
        panel.backgroundColor = .clear
        panel.isOpaque = false
        panel.hasShadow = true
        panel.ignoresMouseEvents = true
        panel.hidesOnDeactivate = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
        return panel
    }

    /// Just below and right of the pointer, kept on the pointer's screen.
    private static func place(_ panel: NSPanel) {
        let pointer = NSEvent.mouseLocation
        let size = panel.frame.size
        var origin = NSPoint(x: pointer.x + 16, y: pointer.y - size.height - 16)
        if let visible = (NSScreen.screens.first { $0.frame.contains(pointer) } ?? NSScreen.main)?.visibleFrame {
            origin.x = min(max(origin.x, visible.minX + 8), visible.maxX - size.width - 8)
            origin.y = min(max(origin.y, visible.minY + 8), visible.maxY - size.height - 8)
        }
        panel.setFrameOrigin(origin)
    }
}

private struct HUDView: View {
    let state: ProcessingHUD.State

    var body: some View {
        HStack(spacing: 10) {
            switch state {
            case .recording:
                Circle().fill(.red).frame(width: 10, height: 10)
                Text("Watching your lips… press the shortcut to stop")
            case .reading:
                ProgressView().controlSize(.small)
                Text("Reading your lips…")
            case .done(let text):
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                Text(text).lineLimit(1)
            case .copied(let text):
                Image(systemName: "doc.on.clipboard.fill").foregroundStyle(.blue)
                Text("Copied to clipboard: \(text)").lineLimit(1)
            case .failed(let message):
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                Text(message).lineLimit(2)
            }
        }
        .font(.system(size: 13, weight: .medium))
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .fixedSize()
    }
}

/// Buttons in a panel that never becomes key still answer the first click.
private final class FirstClickHostingView<Content: View>: NSHostingView<Content> {
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
}

private struct OptionsView: View {
    let lines: [String]
    let pasted: String?
    let note: String
    let onPick: (String) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("You said one of these").font(.system(size: 12, weight: .medium)).foregroundStyle(.secondary)
            ForEach(lines, id: \.self) { line in
                Button { onPick(line) } label: {
                    HStack(spacing: 8) {
                        Image(systemName: line == pasted ? "checkmark.circle.fill" : "circle")
                            .foregroundStyle(line == pasted ? .green : .secondary)
                        Text(line).font(.system(size: 14, weight: .medium))
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 10).padding(.vertical, 7)
                    .frame(minWidth: 280, alignment: .leading)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                }
                .buttonStyle(.plain)
            }
            Text(note).font(.system(size: 11)).foregroundStyle(.secondary)
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .fixedSize()
    }
}
