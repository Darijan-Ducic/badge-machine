import AppKit
import ApplicationServices

struct LeftOptionTap {
    private var armed = false

    mutating func handle(_ event: NSEvent) -> Bool {
        guard event.type == .flagsChanged, UInt32(event.keyCode) == Shortcut.leftOption.keyCode else {
            armed = false
            return false
        }

        // The device flags distinguish left and right Option when both are down.
        let sideFlags = event.modifierFlags.rawValue & 0x60
        let pressed = sideFlags == 0 ? event.modifierFlags.contains(.option) : sideFlags & 0x20 != 0
        let chordModifiers: NSEvent.ModifierFlags = [.command, .control, .shift, .function]
        if pressed {
            armed = sideFlags & 0x40 == 0 && event.modifierFlags.intersection(chordModifiers).isEmpty
            return false
        }

        let tapped = armed && !event.modifierFlags.contains(.option)
        armed = false
        return tapped
    }
}

@MainActor
final class LeftOptionMonitor {
    private var globalMonitor: Any?
    private var localMonitor: Any?
    private var tap = LeftOptionTap()
    var onTap: (() -> Void)?

    func start() -> Bool {
        guard AXIsProcessTrusted() else { stop(); return false }
        if globalMonitor != nil { return true }
        let events: NSEvent.EventTypeMask = [.flagsChanged, .keyDown, .leftMouseDown, .rightMouseDown, .otherMouseDown, .scrollWheel]
        guard let monitor = NSEvent.addGlobalMonitorForEvents(matching: events, handler: { [weak self] event in
            self?.handle(event)
        }) else { return false }
        globalMonitor = monitor
        localMonitor = NSEvent.addLocalMonitorForEvents(matching: events) { [weak self] event in
            self?.handle(event)
            return event
        }
        return true
    }

    private func handle(_ event: NSEvent) {
        if tap.handle(event) { onTap?() }
    }

    func stop() {
        if let globalMonitor { NSEvent.removeMonitor(globalMonitor) }
        if let localMonitor { NSEvent.removeMonitor(localMonitor) }
        globalMonitor = nil
        localMonitor = nil
        tap = LeftOptionTap()
    }

    deinit {
        if let globalMonitor { NSEvent.removeMonitor(globalMonitor) }
        if let localMonitor { NSEvent.removeMonitor(localMonitor) }
    }
}
