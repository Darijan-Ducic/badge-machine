import CoreGraphics
import IOKit.hid

struct FnTap {
    private enum State { case idle, armed, chord }

    private var state = State.idle

    mutating func handle(isFn: Bool, pressed: Bool, modifiersHeld: Bool = false) -> Bool? {
        if !isFn {
            if pressed && state == .armed { state = .chord }
            return nil
        }
        if pressed {
            guard state == .idle else { return nil }
            state = modifiersHeld ? .chord : .armed
            return true
        }
        let tapped = state == .armed
        state = .idle
        return tapped ? false : nil
    }
}

@MainActor
final class FnKeyMonitor {
    private var manager: IOHIDManager?
    private var tap = FnTap()
    // Presses are available for shortcut preview. Only an unchorded release activates it.
    var onChange: ((Bool) -> Void)?

    func start() -> Bool {
        guard CGPreflightListenEventAccess() else {
            stop()
            return false
        }
        if manager != nil { return true }

        let options = IOOptionBits(kIOHIDOptionsTypeNone)
        let manager = IOHIDManagerCreate(kCFAllocatorDefault, options)
        IOHIDManagerSetDeviceMatchingMultiple(manager, [
            [kIOHIDDeviceUsagePageKey: 0x01, kIOHIDDeviceUsageKey: 0x06],
            [kIOHIDDeviceUsagePageKey: 0xFF, kIOHIDDeviceUsageKey: 0x03]
        ] as CFArray)
        // Apple exposes the physical Fn key as a vendor HID element.
        IOHIDManagerSetInputValueMatchingMultiple(manager, [
            [kIOHIDElementUsagePageKey: 0xFF, kIOHIDElementUsageKey: 0x03],
            [kIOHIDElementUsagePageKey: kHIDPage_KeyboardOrKeypad],
            [kIOHIDElementUsagePageKey: kHIDPage_Consumer]
        ] as CFArray)
        IOHIDManagerRegisterInputValueCallback(manager, { context, result, _, value in
            guard let context, result == kIOReturnSuccess else { return }
            MainActor.assumeIsolated {
                Unmanaged<FnKeyMonitor>.fromOpaque(context).takeUnretainedValue().handle(value)
            }
        }, Unmanaged.passUnretained(self).toOpaque())

        IOHIDManagerScheduleWithRunLoop(manager, CFRunLoopGetMain(), CFRunLoopMode.commonModes.rawValue)
        guard IOHIDManagerOpen(manager, options) == kIOReturnSuccess else {
            IOHIDManagerUnscheduleFromRunLoop(manager, CFRunLoopGetMain(), CFRunLoopMode.commonModes.rawValue)
            return false
        }
        self.manager = manager
        return true
    }

    func stop() {
        tap = FnTap()
        guard let manager else { return }
        Self.close(manager)
        self.manager = nil
    }

    private func handle(_ value: IOHIDValue) {
        let element = IOHIDValueGetElement(value)
        let page = IOHIDElementGetUsagePage(element)
        let usage = IOHIDElementGetUsage(element)
        let isFn = page == 0xFF && usage == 0x03
        if page == UInt32(kHIDPage_KeyboardOrKeypad), usage <= UInt32(kHIDUsage_KeyboardErrorUndefined) { return }
        let pressed = IOHIDValueGetIntegerValue(value) != 0
        let chordFlags: CGEventFlags = [.maskCommand, .maskControl, .maskAlternate, .maskShift]
        let modifiersHeld = isFn && pressed && !CGEventSource.flagsState(.combinedSessionState).intersection(chordFlags).isEmpty
        if let change = tap.handle(isFn: isFn, pressed: pressed, modifiersHeld: modifiersHeld) { onChange?(change) }
    }

    deinit {
        if let manager { Self.close(manager) }
    }

    private nonisolated static func close(_ manager: IOHIDManager) {
        IOHIDManagerUnscheduleFromRunLoop(manager, CFRunLoopGetMain(), CFRunLoopMode.commonModes.rawValue)
        IOHIDManagerClose(manager, IOOptionBits(kIOHIDOptionsTypeNone))
    }
}
