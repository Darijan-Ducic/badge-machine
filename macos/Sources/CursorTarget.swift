import AppKit
import ApplicationServices

@MainActor
struct CursorTarget {
    let pid: pid_t
    let element: AXUIElement

    static func current() -> Self? {
        guard let app = NSWorkspace.shared.frontmostApplication,
              app.processIdentifier != ProcessInfo.processInfo.processIdentifier else { return nil }
        let application = AXUIElementCreateApplication(app.processIdentifier)
        var focused: CFTypeRef?
        guard AXUIElementCopyAttributeValue(application, kAXFocusedUIElementAttribute as CFString, &focused) == .success,
              let focused, CFGetTypeID(focused) == AXUIElementGetTypeID() else { return nil }
        let element = focused as! AXUIElement
        var subrole: CFTypeRef?
        AXUIElementCopyAttributeValue(element, kAXSubroleAttribute as CFString, &subrole)
        guard (subrole as? String) != kAXSecureTextFieldSubrole else { return nil }
        var role: CFTypeRef?
        AXUIElementCopyAttributeValue(element, kAXRoleAttribute as CFString, &role)
        let textRoles = [kAXTextFieldRole, kAXTextAreaRole, kAXComboBoxRole]
        var selectedTextIsSettable = DarwinBoolean(false)
        AXUIElementIsAttributeSettable(element, kAXSelectedTextAttribute as CFString, &selectedTextIsSettable)
        guard textRoles.contains(role as? String ?? "") || selectedTextIsSettable.boolValue else { return nil }
        return Self(pid: app.processIdentifier, element: element)
    }

    enum InsertionResult {
        case inserted, pasteRequested, targetUnavailable
    }

    func insert(_ text: String) -> InsertionResult {
        guard let current = Self.current(), current.pid == pid,
              CFEqual(current.element, element) else { return .targetUnavailable }
        if AXUIElementSetAttributeValue(element, kAXSelectedTextAttribute as CFString, text as CFString) == .success {
            return .inserted
        }

        guard let down = CGEvent(keyboardEventSource: nil, virtualKey: 9, keyDown: true),
              let up = CGEvent(keyboardEventSource: nil, virtualKey: 9, keyDown: false) else { return .targetUnavailable }
        let clipboard = NSPasteboard.general
        let previous = (clipboard.pasteboardItems ?? []).map { item in
            let copy = NSPasteboardItem()
            for type in item.types {
                if let data = item.data(forType: type) { copy.setData(data, forType: type) }
            }
            return copy
        }
        guard let pasteTarget = Self.current(), pasteTarget.pid == pid,
              CFEqual(pasteTarget.element, element) else { return .targetUnavailable }
        clipboard.clearContents()
        clipboard.setString(text, forType: .string)
        let inserted = clipboard.changeCount
        for event in [down, up] {
            event.flags = .maskCommand
            event.postToPid(pid)
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
            guard clipboard.changeCount == inserted else { return }
            clipboard.clearContents()
            if !previous.isEmpty { clipboard.writeObjects(previous) }
        }
        return .pasteRequested
    }
}
