import AppKit
import ApplicationServices
import AVFoundation
import SwiftUI

@MainActor
final class Studio: ObservableObject {
    enum Mode: String, CaseIterable {
        case whisper = "Silent Whisper"
        case cursor = "Move Cursor"

        var preferenceKey: String {
            switch self {
            case .whisper: "recordingShortcut"
            case .cursor: "cursorShortcut"
            }
        }

        var description: String {
            switch self {
            case .whisper: "Move your lips silently and type at the text cursor."
            case .cursor: "Point with your right index and middle fingers together. Close them to click."
            }
        }
    }

    enum Phase { case ready, starting, recording, processing, handStarting, handCursor, handStopping }

    @Published private(set) var phase: Phase = .ready
    @Published private(set) var status = ""
    @Published private(set) var shortcuts: [Mode: Shortcut] = [:]
    @Published private(set) var editingMode: Mode?
    var choosing: Bool { editingMode != nil }
    @Published private(set) var pendingShortcut: Shortcut?
    private let camera = Capture()
    private let handCamera = HandCursor()
    private var handPointer = HandPointer()
    private var handRun: UUID?
    private let hotkeys: [Mode: GlobalShortcut] = [
        .whisper: GlobalShortcut(signature: 0x4241494A),
        .cursor: GlobalShortcut(signature: 0x4241494B)
    ]
    private let fnMonitor = FnKeyMonitor()
    private let modifierTapMonitor = ModifierTapMonitor()
    @Published private(set) var modifierMonitoringReady = false
    private var fnArmed = false
    @Published private(set) var fnMonitoringReady = false
    private let gemini: Gemini?
    private let muse: MuseSpark?
    private var target: CursorTarget?

    init() {
        let keys = APIKeys()
        gemini = keys.gemini.map(Gemini.init(key:))
        muse = keys.meta.map(MuseSpark.init(key:))
        modifierTapMonitor.onTap = { [weak self] shortcut in
            guard let self, let mode = Mode.allCases.first(where: {
                self.shortcuts[$0]?.isStandaloneModifier == true && self.shortcuts[$0]?.keyCode == shortcut.keyCode
            }) else { return }
            self.activate(mode)
        }
        fnMonitor.onChange = { [weak self] pressed in self?.handleFn(pressed: pressed) }
        for mode in Mode.allCases {
            hotkeys[mode]?.onRelease = { [weak self] in self?.activate(mode) }
            guard let data = UserDefaults.standard.data(forKey: mode.preferenceKey),
                  let saved = try? JSONDecoder().decode(Shortcut.self, from: data) else { continue }
            shortcuts[mode] = saved
            restoreShortcut(for: mode)
        }
        resumeShortcutMonitoring()
    }

    private var fnMode: Mode? {
        Mode.allCases.first { shortcuts[$0]?.isFn == true }
    }

    private var hasModifierShortcut: Bool {
        shortcuts.values.contains(where: \.isStandaloneModifier)
    }

    var needsModifierSetup: Bool { !choosing && hasModifierShortcut && !modifierMonitoringReady }
    var needsFnSetup: Bool { !choosing && fnMode != nil && !fnMonitoringReady }

    private func activate(_ mode: Mode) {
        switch mode {
        case .whisper: toggle()
        case .cursor: toggleHandCursor()
        }
    }

    private func restoreShortcut(for mode: Mode) {
        guard let shortcut = shortcuts[mode], !shortcut.usesEventMonitor else { return }
        if hotkeys[mode]?.register(shortcut) != true {
            status = "The shortcut for \(mode.rawValue) is unavailable. Choose another."
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

    var canToggleHandCursor: Bool { !choosing && (phase == .ready || phase == .handCursor) }

    func toggleHandCursor() {
        guard canToggleHandCursor else { return }
        if phase == .handCursor {
            phase = .handStopping
            handRun = nil
            status = "Stopping Move Cursor…"
            handCamera.stop { [weak self] in
                DispatchQueue.main.async { [weak self] in
                    self?.phase = .ready
                    self?.status = ""
                }
            }
            return
        }
        guard AXIsProcessTrusted() else {
            requestInsertionAccess(message: "")
            return
        }
        phase = .handStarting
        status = "Starting Move Cursor…"
        Task { await startHandCursor() }
    }

    private func startHandCursor() async {
        guard await AVCaptureDevice.requestAccess(for: .video) else {
            phase = .ready
            status = "Allow Camera access in System Settings."
            return
        }
        let run = UUID()
        handRun = run
        handPointer = HandPointer()
        handCamera.start(onGesture: { [weak self] gesture, timestamp in
            DispatchQueue.main.async { [weak self] in
                guard let self, self.handRun == run, self.phase == .handCursor else { return }
                guard ProcessInfo.processInfo.systemUptime - timestamp < 0.2 else {
                    self.handPointer = HandPointer()
                    return
                }
                let bounds = CGDisplayBounds(CGMainDisplayID())
                self.handPointer.event(for: gesture, at: timestamp, in: bounds)?.post()
            }
        }, onStart: { [weak self] running in
            DispatchQueue.main.async { [weak self] in
                guard let self, self.handRun == run else { return }
                self.phase = running ? .handCursor : .ready
                self.status = running ? "Move Cursor is on. Point with two fingers together." : "Could not start the camera."
                if !running { self.handRun = nil }
            }
        }, onFailure: { [weak self] in
            DispatchQueue.main.async { [weak self] in
                guard let self, self.handRun == run else { return }
                self.handRun = nil
                self.phase = .ready
                self.status = "The camera stopped. Restart Move Cursor to try again."
            }
        })
    }

    func chooseShortcut(for mode: Mode) {
        guard phase == .ready else { return }
        hotkeys.values.forEach { $0.unregister() }
        fnArmed = false
        editingMode = mode
        status = ""
        resumeShortcutMonitoring()
    }

    func cancelChoosing() {
        editingMode = nil
        pendingShortcut = nil
        status = ""
        resumeShortcutMonitoring()
        for mode in Mode.allCases { restoreShortcut(for: mode) }
    }

    func recordShortcut(_ event: NSEvent) {
        guard !event.isARepeat else { return }
        guard let choice = Shortcut(event: event) else {
            status = "Press a key or a keyboard shortcut."
            return
        }
        pendingShortcut = choice
        status = ""
    }

    func previewFn() { previewModifier(.fn) }

    func previewModifier(_ shortcut: Shortcut) {
        guard choosing else { return }
        pendingShortcut = shortcut
        status = ""
    }

    func resumeShortcutMonitoring() {
        if !choosing && hasModifierShortcut {
            modifierMonitoringReady = modifierTapMonitor.start()
        } else {
            modifierTapMonitor.stop()
            modifierMonitoringReady = false
        }
        if choosing || fnMode != nil {
            fnMonitoringReady = fnMonitor.start()
        } else {
            fnMonitor.stop()
            fnMonitoringReady = false
        }
    }

    func openInputMonitoring() {
        if !CGPreflightListenEventAccess() { _ = CGRequestListenEventAccess() }
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent")!
        NSWorkspace.shared.open(url)
    }

    private func handleFn(pressed: Bool) {
        if choosing {
            if pressed && NSApp.isActive { previewFn() }
            return
        }
        guard let mode = fnMode else { return }
        if pressed {
            fnArmed = true
        } else if fnArmed {
            fnArmed = false
            activate(mode)
        }
    }

    func confirmShortcut() {
        guard let mode = editingMode, let choice = pendingShortcut else { return }
        if let conflict = shortcuts.first(where: {
            $0.key != mode && $0.value.keyCode == choice.keyCode && $0.value.modifiers == choice.modifiers
        }) {
            status = "That shortcut is already set for \(conflict.key.rawValue)."
            return
        }
        if !choice.usesEventMonitor && hotkeys[mode]?.register(choice) != true {
            status = "That shortcut is already in use. Try another."
            return
        }

        shortcuts[mode] = choice
        UserDefaults.standard.set(try? JSONEncoder().encode(choice), forKey: mode.preferenceKey)
        editingMode = nil
        pendingShortcut = nil
        status = ""
        resumeShortcutMonitoring()
        for other in Mode.allCases where other != mode { restoreShortcut(for: other) }
    }

    func openModifierAccessibility() {
        requestInsertionAccess(message: "")
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!
        NSWorkspace.shared.open(url)
    }

    private func requestInsertionAccess(message: String = "Allow Accessibility access to type in other apps.") {
        if AXIsProcessTrusted() {
            status = ""
        } else {
            status = message
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
        case .processing, .handStarting, .handCursor, .handStopping:
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
