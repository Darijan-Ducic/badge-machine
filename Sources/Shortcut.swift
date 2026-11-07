import AppKit
import Carbon

struct Shortcut: Codable {
    let keyCode: UInt32
    let modifiers: UInt32
    let label: String

    init?(event: NSEvent) {
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
            36: "Return", 48: "Tab", 49: "Space", 51: "Delete",
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
        let key = namedKeys[event.keyCode] ?? event.charactersIgnoringModifiers?.uppercased() ?? ""
        guard !key.isEmpty,
              namedKeys[event.keyCode] != nil || key.unicodeScalars.allSatisfy({
                  CharacterSet.alphanumerics.union(.punctuationCharacters).union(.symbols).contains($0)
                      && $0.value < 0xF700
              }) else { return nil }
        self.keyCode = UInt32(event.keyCode)
        self.modifiers = modifiers
        self.label = label + key
    }
}

@MainActor
final class GlobalShortcut {
    private var handler: EventHandlerRef?
    private var registration: EventHotKeyRef?
    private var nextID: UInt32 = 0
    private var activeID: UInt32 = 0
    var onRelease: (() -> Void)?

    init() {
        var type = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased))
        InstallEventHandler(GetApplicationEventTarget(), { _, event, context in
            guard let event, let context else { return OSStatus(eventNotHandledErr) }
            var id = EventHotKeyID()
            guard GetEventParameter(event, EventParamName(kEventParamDirectObject), EventParamType(typeEventHotKeyID),
                                    nil, MemoryLayout<EventHotKeyID>.size, nil, &id) == noErr,
                  id.signature == 0x4241494A else { return OSStatus(eventNotHandledErr) }
            let owner = Unmanaged<GlobalShortcut>.fromOpaque(context).takeUnretainedValue()
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
        let id = EventHotKeyID(signature: 0x4241494A, id: nextID)
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
