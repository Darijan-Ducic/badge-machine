import AppKit
import Carbon

struct Shortcut: Codable {
    let keyCode: UInt32
    let modifiers: UInt32
    let label: String

    static let fnKeyCode = UInt32(kVK_Function)
    static let fn = Self(keyCode: fnKeyCode, modifiers: 0, label: "Fn")
    var isStandaloneModifier: Bool { modifiers == 0 && ModifierKey.byCode[keyCode] != nil }
    var usesEventMonitor: Bool { isFn || isStandaloneModifier }

    init(modifierKeyCode: UInt32, label: String) {
        self.init(keyCode: modifierKeyCode, modifiers: 0, label: label)
    }

    var isFn: Bool { keyCode == Self.fnKeyCode && modifiers == 0 }

    static func pressedModifier(_ event: NSEvent) -> Self? {
        guard event.type == .flagsChanged else { return nil }
        let code = UInt32(event.keyCode)
        if code == fnKeyCode, event.modifierFlags.contains(.function) { return .fn }
        guard let key = ModifierKey.byCode[code], event.modifierFlags.contains(key.flag) else { return nil }
        let deviceFlags = event.modifierFlags.rawValue & ModifierKey.deviceMasks
        guard deviceFlags == 0 || deviceFlags & key.deviceMask != 0 else { return nil }
        return Self(modifierKeyCode: code, label: key.label)
    }

    private init(keyCode: UInt32, modifiers: UInt32, label: String) {
        self.keyCode = keyCode
        self.modifiers = modifiers
        self.label = label
    }

    init?(event: NSEvent) {
        guard event.type == .keyDown else { return nil }
        let flags = event.modifierFlags
        var modifiers: UInt32 = 0
        var label = ""
        for (flag, carbon, symbol) in [
            (NSEvent.ModifierFlags.control, controlKey, "⌃"),
            (.option, optionKey, "⌥"), (.shift, shiftKey, "⇧"), (.command, cmdKey, "⌘")
        ] where flags.contains(flag) {
            modifiers |= UInt32(carbon)
            label += symbol
        }
        let namedKeys: [UInt16: String] = [
            36: "Return", 48: "Tab", 49: "Space", 51: "Delete", 53: "Escape",
            71: "Keypad Clear", 76: "Keypad Enter", 114: "Help",
            115: "Home", 116: "Page Up", 117: "Forward Delete", 119: "End", 121: "Page Down",
            123: "←", 124: "→", 125: "↓", 126: "↑"
        ]
        let functionKeys: [UInt16: String] = [
            122: "F1", 120: "F2", 99: "F3", 118: "F4", 96: "F5", 97: "F6",
            98: "F7", 100: "F8", 101: "F9", 109: "F10", 103: "F11", 111: "F12",
            105: "F13", 107: "F14", 113: "F15", 106: "F16", 64: "F17",
            79: "F18", 80: "F19", 90: "F20"
        ]
        if let key = functionKeys[event.keyCode] {
            self.keyCode = UInt32(event.keyCode)
            self.modifiers = modifiers
            self.label = label + key
            return
        }
        let character = event.charactersIgnoringModifiers?.uppercased() ?? ""
        let printable = !character.isEmpty && character.unicodeScalars.allSatisfy {
            !CharacterSet.controlCharacters.contains($0) && !(0xF700...0xF8FF).contains($0.value)
        }
        let key = namedKeys[event.keyCode] ?? (printable ? character : "Key \(event.keyCode)")
        self.keyCode = UInt32(event.keyCode)
        self.modifiers = modifiers
        self.label = label + key
    }
}

@MainActor
final class GlobalShortcut {
    private let signature: OSType
    private var handler: EventHandlerRef?
    private var registration: EventHotKeyRef?
    private var nextID: UInt32 = 0
    private var activeID: UInt32 = 0
    var onRelease: (() -> Void)?

    init(signature: OSType) {
        self.signature = signature
        var type = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased))
        InstallEventHandler(GetApplicationEventTarget(), { _, event, context in
            guard let event, let context else { return OSStatus(eventNotHandledErr) }
            var id = EventHotKeyID()
            guard GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                                    nil, MemoryLayout<EventHotKeyID>.size, nil, &id) == noErr else { return OSStatus(eventNotHandledErr) }
            let owner = Unmanaged<GlobalShortcut>.fromOpaque(context).takeUnretainedValue()
            guard id.signature == owner.signature else { return OSStatus(eventNotHandledErr) }
            let releasedID = id.id
            DispatchQueue.main.async {
                if owner.activeID == releasedID { owner.onRelease?() }
            }
            return noErr
        }, 1, &type, Unmanaged.passUnretained(self).toOpaque(), &handler)
    }

    func register(_ shortcut: Shortcut) -> Bool {
        guard handler != nil else { return false }
        nextID &+= 1
        var candidate: EventHotKeyRef?
        let id = EventHotKeyID(signature: signature, id: nextID)
        guard RegisterEventHotKey(shortcut.keyCode, shortcut.modifiers, id,
                                  GetApplicationEventTarget(), 0, &candidate) == noErr,
              let candidate else { return false }
        if let registration { UnregisterEventHotKey(registration) }
        registration = candidate
        activeID = id.id
        return true
    }

    func unregister() {
        if let registration { UnregisterEventHotKey(registration) }
        registration = nil
        activeID = 0
    }

    deinit {
        if let registration { UnregisterEventHotKey(registration) }
        if let handler { RemoveEventHandler(handler) }
    }
}
