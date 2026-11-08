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
