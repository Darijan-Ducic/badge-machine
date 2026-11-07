import AppKit
import ApplicationServices
import AVFoundation
import SwiftUI

@MainActor
final class Studio: ObservableObject {
    enum Phase { case ready, starting, recording, processing }

    @Published private(set) var phase: Phase = .ready
    @Published private(set) var status = ""
    @Published private(set) var shortcut: Shortcut?
    @Published private(set) var choosing = false
    @Published private(set) var pendingShortcut: Shortcut?
    private let camera = Capture()
    private let hotkey = GlobalShortcut()
    private let fnMonitor = FnKeyMonitor()
    private var fnArmed = false
    private var requestedInputMonitoring = false
    private let gemini: Gemini?
    private let muse: MuseSpark?
    private var target: CursorTarget?

    init() {
        let keys = APIKeys()
        gemini = keys.gemini.map(Gemini.init(key:))
        muse = keys.meta.map(MuseSpark.init(key:))
        hotkey.onRelease = { [weak self] in self?.toggle() }
        fnMonitor.onChange = { [weak self] pressed in self?.handleFn(pressed: pressed) }
        if let data = UserDefaults.standard.data(forKey: "recordingShortcut"),
           let saved = try? JSONDecoder().decode(Shortcut.self, from: data) {
            if saved.isFn {
                shortcut = saved
                resumeFnMonitoring()
            } else if hotkey.register(saved) {
                shortcut = saved
            } else {
                status = "Choose another shortcut. Your saved shortcut is unavailable."
            }
        }
    }

    func requestCamera() async {
        if AVCaptureDevice.authorizationStatus(for: .video) == .notDetermined {
            _ = await AVCaptureDevice.requestAccess(for: .video)
        }
        if AVCaptureDevice.authorizationStatus(for: .video) != .authorized {
            status = "Allow Camera access in System Settings."
        }
    }

    func chooseShortcut() {
        hotkey.unregister()
        fnArmed = false
        choosing = true
        status = ""
        if !fnMonitor.start() && !requestedInputMonitoring {
            requestedInputMonitoring = true
            if !CGRequestListenEventAccess() {
                status = "Allow Input Monitoring to capture Fn."
            }
        }
    }

    func cancelChoosing() {
        choosing = false
        pendingShortcut = nil
        status = ""
        if shortcut?.isFn != true {
            fnMonitor.stop()
            if let shortcut, !hotkey.register(shortcut) {
                status = "Your saved shortcut is unavailable. Choose another."
            }
        }
    }

    func recordShortcut(_ event: NSEvent) {
        if event.keyCode == 53 { cancelChoosing(); return }
        guard !event.isARepeat else { return }
        guard let choice = Shortcut(event: event) else {
            status = "Press a key or a keyboard shortcut."
            return
        }
        pendingShortcut = choice
        status = ""
    }

    func previewFn() {
        guard choosing else { return }
        pendingShortcut = .fn
        status = "For Fn without emoji, set the Globe key to Do Nothing in Keyboard settings."
    }

    func resumeFnMonitoring() {
        guard choosing || shortcut?.isFn == true else { return }
        if fnMonitor.start() {
            if status == "Allow Input Monitoring to capture Fn." { status = "" }
        } else {
            status = "Allow Input Monitoring to capture Fn."
        }
    }

    private func handleFn(pressed: Bool) {
        if choosing {
            if pressed && NSApp.isActive { previewFn() }
            return
        }
        guard shortcut?.isFn == true else { return }
        if pressed {
            fnArmed = true
        } else if fnArmed {
            fnArmed = false
            toggle()
        }
    }

    func confirmShortcut() {
        guard let choice = pendingShortcut else { return }
        if choice.isFn {
            guard fnMonitor.start() else {
                _ = CGRequestListenEventAccess()
                status = "Allow Input Monitoring to use Fn, then reopen Baijiu Machine."
                return
            }
        } else {
            guard hotkey.register(choice) else {
                status = "That shortcut is already in use. Try another."
                return
            }
            fnMonitor.stop()
        }
        shortcut = choice
        UserDefaults.standard.set(try? JSONEncoder().encode(choice), forKey: "recordingShortcut")
        choosing = false
        pendingShortcut = nil
        requestInsertionAccess()
    }

    private func requestInsertionAccess() {
        if AXIsProcessTrusted() {
            status = ""
        } else {
            status = "Allow Accessibility access to type in other apps."
            let options = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
            _ = AXIsProcessTrustedWithOptions(options)
        }
    }

    private func toggle() {
        guard !choosing else { return }
        switch phase {
        case .starting, .recording:
            let cameraStarted = phase == .recording
            target = CursorTarget.current()
            phase = .processing
            status = "Transcribing…"
            if cameraStarted { camera.stop() }
        case .ready:
            guard AVCaptureDevice.authorizationStatus(for: .video) == .authorized else {
                status = "Allow Camera access in System Settings."
                return
            }
            guard AXIsProcessTrusted() else { requestInsertionAccess(); return }
            guard gemini != nil else { status = "Gemini is not configured."; return }
            phase = .starting
            status = "Starting camera…"
            camera.start(onStart: { [weak self] in
                Task { @MainActor in
                    guard let self else { return }
                    if self.phase == .processing { self.camera.stop(); return }
                    guard self.phase == .starting else { return }
                    self.phase = .recording
                    self.status = "Recording…"
                }
            }, onFinish: { [weak self] result in
                Task { @MainActor in await self?.finish(result) }
            })
        case .processing:
            break
        }
    }

    private func finish(_ result: Result<URL, Error>) async {
        defer { phase = .ready; target = nil }
        do {
            let file = try result.get()
            defer { try? FileManager.default.removeItem(at: file) }
            guard let target else { status = "Place the cursor in a text field before stopping."; return }
            guard let gemini else { return }
            guard let draft = try await gemini.transcribe(file) else {
                status = "Could not read the phrase. Try again."
                return
            }
            let revision = if let muse { try? await muse.refine(draft) } else { nil as MuseSpark.Revision? }
            status = target.insert(revision?.text ?? draft) ? "Text inserted." : "The original text field is no longer active."
        } catch {
            status = error.localizedDescription
        }
    }
}
