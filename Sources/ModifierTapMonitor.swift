import AppKit
import ApplicationServices

import Carbon
import IOKit.hidsystem

struct ModifierKey {
    let label: String
    let flag: NSEvent.ModifierFlags
    let deviceMask: UInt

    static let byCode: [UInt32: Self] = [
        UInt32(kVK_Option): Self(label: "Left Option", flag: .option, deviceMask: UInt(NX_DEVICELALTKEYMASK)),
        UInt32(kVK_RightOption): Self(label: "Right Option", flag: .option, deviceMask: UInt(NX_DEVICERALTKEYMASK)),
        UInt32(kVK_Control): Self(label: "Left Control", flag: .control, deviceMask: UInt(NX_DEVICELCTLKEYMASK)),
        UInt32(kVK_RightControl): Self(label: "Right Control", flag: .control, deviceMask: UInt(NX_DEVICERCTLKEYMASK)),
        UInt32(kVK_Command): Self(label: "Left Command", flag: .command, deviceMask: UInt(NX_DEVICELCMDKEYMASK)),
        UInt32(kVK_RightCommand): Self(label: "Right Command", flag: .command, deviceMask: UInt(NX_DEVICERCMDKEYMASK)),
        UInt32(kVK_Shift): Self(label: "Left Shift", flag: .shift, deviceMask: UInt(NX_DEVICELSHIFTKEYMASK)),
        UInt32(kVK_RightShift): Self(label: "Right Shift", flag: .shift, deviceMask: UInt(NX_DEVICERSHIFTKEYMASK))
    ]

    static let deviceMasks = byCode.values.reduce(UInt(0)) { $0 | $1.deviceMask }
}

struct ModifierTap {
    private var armedKey: UInt32?

    mutating func handle(_ event: NSEvent) -> Shortcut? {
        let previousKey = armedKey
        armedKey = nil
        guard event.type == .flagsChanged,
              let key = ModifierKey.byCode[UInt32(event.keyCode)] else { return nil }

        let code = UInt32(event.keyCode)
        let chordFlags: NSEvent.ModifierFlags = [.command, .control, .option, .shift, .function]
        let activeFlags = event.modifierFlags.intersection(chordFlags)
        let deviceFlags = event.modifierFlags.rawValue & ModifierKey.deviceMasks
        if activeFlags == key.flag && (deviceFlags == 0 || deviceFlags == key.deviceMask) {
            armedKey = code
            return nil
        }
        guard activeFlags.isEmpty, previousKey == code else { return nil }
        return Shortcut(modifierKeyCode: code, label: key.label)
    }
}

@MainActor
final class ModifierTapMonitor {
    private var globalMonitor: Any?
    private var localMonitor: Any?
    private var tap = ModifierTap()
    var onTap: ((Shortcut) -> Void)?

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
        if let shortcut = tap.handle(event) { onTap?(shortcut) }
    }

    func stop() {
        if let globalMonitor { NSEvent.removeMonitor(globalMonitor) }
        if let localMonitor { NSEvent.removeMonitor(localMonitor) }
        globalMonitor = nil
        localMonitor = nil
        tap = ModifierTap()
    }

    deinit {
        if let globalMonitor { NSEvent.removeMonitor(globalMonitor) }
        if let localMonitor { NSEvent.removeMonitor(localMonitor) }
    }
}
