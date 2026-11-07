import CoreGraphics
import IOKit.hid

@MainActor
final class FnKeyMonitor {
    private var manager: IOHIDManager?
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
        IOHIDManagerSetInputValueMatching(manager, [
            kIOHIDElementUsagePageKey: 0xFF,
            kIOHIDElementUsageKey: 0x03
        ] as CFDictionary)
        IOHIDManagerRegisterInputValueCallback(manager, { context, result, _, value in
            guard let context, result == kIOReturnSuccess else { return }
            let pressed = IOHIDValueGetIntegerValue(value) != 0
            MainActor.assumeIsolated {
                Unmanaged<FnKeyMonitor>.fromOpaque(context).takeUnretainedValue().onChange?(pressed)
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
        guard let manager else { return }
        Self.close(manager)
        self.manager = nil
    }

    deinit {
        if let manager { Self.close(manager) }
    }

    private nonisolated static func close(_ manager: IOHIDManager) {
        IOHIDManagerUnscheduleFromRunLoop(manager, CFRunLoopGetMain(), CFRunLoopMode.commonModes.rawValue)
        IOHIDManagerClose(manager, IOOptionBits(kIOHIDOptionsTypeNone))
    }
}
