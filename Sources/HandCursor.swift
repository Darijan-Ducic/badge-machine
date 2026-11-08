import AVFoundation
import CoreGraphics
import os
import Vision

final class HandCursor: @unchecked Sendable {
    static let maximumResultAge: TimeInterval = 0.5
    private static let logger = Logger(subsystem: "com.darijanducic.baijiumachine", category: "HandCapture")
    private static let maximumRetries = 2
    private let queue = DispatchQueue(label: "baijiu.hand-cursor", qos: .userInteractive)

    private struct Run {
        let id = UUID()
        let onGesture: (HandGesture, TimeInterval) -> Void
        let onFailure: () -> Void
        var retries = 0
    }

    private struct Attempt {
        let id: UUID
        let session: AVCaptureSession
        let output: AVCaptureVideoDataOutput
        let tracker: HandFrameTracker
        let observers: [NSObjectProtocol]
        var startedAt: TimeInterval
        var lastMetricsAt: TimeInterval?
        var reportedFirstFrame = false
        var lastUsablePoseAt: TimeInterval?
        var reportedPoseGap = false
    }

    private var run: Run?
    private var attempt: Attempt?
    private var watchdog: DispatchSourceTimer?

    func start(onGesture: @escaping (HandGesture, TimeInterval) -> Void, onStart: @escaping (Bool) -> Void,
               onFailure: @escaping () -> Void) {
        queue.async { [self] in
            guard run == nil else { onStart(false); return }
            run = Run(onGesture: onGesture, onFailure: onFailure)
            let running = startAttempt()
            if !running { run = nil }
            onStart(running)
        }
    }

    func stop(onStop: @escaping () -> Void) {
        queue.async { [self] in
            run = nil
            releaseAttempt()
            Self.logger.notice("Move Cursor capture stopped by user")
            onStop()
        }
    }

    private func startAttempt() -> Bool {
        guard let run else { return false }
        let session = AVCaptureSession()
        let output = AVCaptureVideoDataOutput()
        let id = UUID()
        let start = ProcessInfo.processInfo.systemUptime
        let number = run.retries + 1
        Self.logger.notice("Starting Move Cursor capture attempt \(number, privacy: .public)")
        guard configure(session: session, output: output) else { return false }

        let tracker = HandFrameTracker(callbackQueue: queue) { [weak self] gesture, timestamp, succeeded in
            guard let self, self.attempt?.id == id else { return }
            if succeeded && self.attempt?.reportedFirstFrame == false {
                self.attempt?.reportedFirstFrame = true
                let elapsed = Int((ProcessInfo.processInfo.systemUptime - start) * 1000)
                Self.logger.notice("First Vision result for capture attempt \(number, privacy: .public) after \(elapsed, privacy: .public) ms")
            }
            self.recordPose(gesture, at: timestamp)
            self.run?.onGesture(gesture, timestamp)
        }
        output.setSampleBufferDelegate(tracker, queue: DispatchQueue(label: "baijiu.hand-cursor.frames", qos: .userInteractive))
        let observers = [AVCaptureSession.runtimeErrorNotification, AVCaptureSession.didStopRunningNotification,
                         AVCaptureSession.wasInterruptedNotification].map { name in
            NotificationCenter.default.addObserver(forName: name, object: session, queue: nil) { [weak self] notification in
                let error = notification.userInfo?[AVCaptureSessionErrorKey] as? NSError
                let reason = error.map { "\(name.rawValue) (\($0.domain) \($0.code))" } ?? name.rawValue
                Self.logger.notice("Move Cursor capture event: \(reason, privacy: .public)")
                self?.queue.async { [weak self] in self?.recover(attemptID: id, reason: reason) }
            }
        }
        attempt = Attempt(id: id, session: session, output: output, tracker: tracker, observers: observers, startedAt: start)
        session.startRunning()
        guard session.isRunning else {
            Self.logger.error("Capture attempt \(number, privacy: .public) failed to start")
            releaseAttempt()
            return false
        }
        attempt?.startedAt = ProcessInfo.processInfo.systemUptime
        tracker.setCaptureClock(session.synchronizationClock)
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + .milliseconds(500), repeating: .milliseconds(500), leeway: .milliseconds(100))
        timer.setEventHandler { [weak self] in self?.checkProgress(attemptID: id) }
        watchdog = timer
        timer.activate()
        return true
    }

    private func recordPose(_ gesture: HandGesture, at timestamp: TimeInterval) {
        switch gesture {
        case .extended, .pinching:
            if let attempt, attempt.reportedPoseGap, let lastPose = attempt.lastUsablePoseAt {
                let elapsed = Int((timestamp - lastPose) * 1000)
                Self.logger.notice("Usable Move Cursor pose returned after \(elapsed, privacy: .public) ms")
            }
            attempt?.lastUsablePoseAt = timestamp
            attempt?.reportedPoseGap = false
        case .other, .lost:
            break
        }
    }

    private func checkProgress(attemptID: UUID) {
        guard let attempt, attempt.id == attemptID else { return }
        let progress = attempt.tracker.progress
        let now = ProcessInfo.processInfo.systemUptime
        let metricsDuration = now - (attempt.lastMetricsAt ?? attempt.startedAt)
        if metricsDuration >= 15 {
            self.attempt?.lastMetricsAt = now
            let metrics = attempt.tracker.takeMetrics()
            Self.logger.notice("Move Cursor performance: \(metrics.summary(over: metricsDuration), privacy: .public)")
        }
        let silence = now - (progress.lastSuccess ?? attempt.startedAt)
        if silence < 1, let staleSince = progress.staleSince, now - staleSince >= 3 {
            recover(attemptID: attemptID, reason: "completed results exceeded \(Int(Self.maximumResultAge * 1000)) ms age for \(Int((now - staleSince) * 1000)) ms")
            return
        }
        if silence < 1, let lastPose = attempt.lastUsablePoseAt, now - lastPose >= 3, !attempt.reportedPoseGap {
            self.attempt?.reportedPoseGap = true
            Self.logger.notice("Vision is completing frames without a usable Move Cursor pose")
        }
        if let run, run.retries > 0, let stableSince = progress.stableSince,
           silence < 1, now - stableSince >= 15 {
            self.run?.retries = 0
            Self.logger.notice("Move Cursor capture stabilized and restored its recovery budget")
        }
        guard silence >= (progress.lastSuccess == nil ? 4 : 3) else { return }
        let reason: String
        if let sample = progress.lastSample, now - sample < 1 {
            reason = progress.lastError.map { "Vision errors (\($0))" } ?? "Vision made no progress"
        } else {
            reason = "camera delivered no recent frames"
        }
        recover(attemptID: attemptID, reason: "\(reason) for \(Int(silence * 1000)) ms")
    }

    private func recover(attemptID: UUID, reason: String) {
        guard let run, attempt?.id == attemptID else { return }
        Self.logger.warning("Move Cursor capture lost progress: \(reason, privacy: .public)")
        run.onGesture(.lost, ProcessInfo.processInfo.systemUptime)
        releaseAttempt()
        retry(runID: run.id)
    }

    private func retry(runID: UUID) {
        guard let run, run.id == runID else { return }
        guard run.retries < Self.maximumRetries else {
            Self.logger.error("Move Cursor capture exhausted automatic recovery")
            self.run = nil
            run.onFailure()
            return
        }
        self.run?.retries += 1
        queue.asyncAfter(deadline: .now() + .milliseconds(750)) { [weak self] in
            guard let self, self.run?.id == runID else { return }
            if !self.startAttempt() { self.retry(runID: runID) }
        }
    }

    private func releaseAttempt() {
        guard let attempt else { return }
        self.attempt = nil
        watchdog?.cancel()
        watchdog = nil
        attempt.observers.forEach { NotificationCenter.default.removeObserver($0) }
        attempt.tracker.stop()
        attempt.output.setSampleBufferDelegate(nil, queue: nil)
        if attempt.session.isRunning { attempt.session.stopRunning() }
        attempt.session.beginConfiguration()
        attempt.session.removeOutput(attempt.output)
        for input in attempt.session.inputs { attempt.session.removeInput(input) }
        attempt.session.commitConfiguration()
    }

    private func configure(session: AVCaptureSession, output: AVCaptureVideoDataOutput) -> Bool {
        guard let camera = AVCaptureDevice.default(for: .video) else {
            Self.logger.error("No camera is available for Move Cursor")
            return false
        }
        let input: AVCaptureDeviceInput
        do {
            input = try AVCaptureDeviceInput(device: camera)
        } catch let error as NSError {
            Self.logger.error("Could not open Move Cursor camera: \(error.domain, privacy: .public) \(error.code, privacy: .public)")
            return false
        }
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        guard session.canAddInput(input), session.canAddOutput(output) else {
            Self.logger.error("Could not configure Move Cursor camera input or output")
            return false
        }
        if session.canSetSessionPreset(.vga640x480) { session.sessionPreset = .vga640x480 }
        session.addInput(input)
        session.addOutput(output)
        if let connection = output.connection(with: .video), connection.isVideoMirroringSupported {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = false
        }
        output.alwaysDiscardsLateVideoFrames = true
        return true
    }

    static func gesture(in hands: [VNHumanHandPoseObservation]) -> HandGesture {
        let rightHands = hands.filter { $0.chirality == .right }
        guard rightHands.count == 1, let hand = rightHands.first,
              let points = try? hand.recognizedPoints(.all) else { return .lost }
        return gesture { joint in
            guard let value = points[joint], value.confidence >= 0.55 else { return nil }
            return value.location
        }
    }

    static func gesture(from point: (VNHumanHandPoseObservation.JointName) -> CGPoint?) -> HandGesture {
        guard let indexBase = point(.indexMCP), let indexJoint = point(.indexPIP), let indexTip = point(.indexTip),
              let middleBase = point(.middleMCP), let middleJoint = point(.middlePIP),
              let middleTip = point(.middleTip) else { return .other }
        func span(to edge: CGPoint) -> CGFloat { hypot(edge.x - indexBase.x, edge.y - indexBase.y) }
        let measuredWidth = point(.littleMCP).map(span)
        guard let palmWidth = measuredWidth ?? point(.ringMCP).map({ span(to: $0) * 1.5 }) else { return .other }
        // A partial palm can still steer and scroll. A click requires the measured span.
        let allowsClick = measuredWidth != nil
        return HandGesture.classify(
            index: Finger(base: indexBase, joint: indexJoint, tip: indexTip),
            middle: Finger(base: middleBase, joint: middleJoint, tip: middleTip),
            thumb: allowsClick ? point(.thumbTip) : nil,
            palmWidth: palmWidth, allowsClick: allowsClick
        )
    }
}

struct Finger {
    let base: CGPoint
    let joint: CGPoint
    let tip: CGPoint

    var reach: CGFloat { hypot(tip.x - base.x, tip.y - base.y) }

    var straightness: CGFloat? {
        let a = CGPoint(x: joint.x - base.x, y: joint.y - base.y)
        let b = CGPoint(x: tip.x - joint.x, y: tip.y - joint.y)
        let length = hypot(a.x, a.y) * hypot(b.x, b.y)
        guard length > 0.00001 else { return nil }
        return (a.x * b.x + a.y * b.y) / length
    }
}

enum HandGesture {
    static let clickDistance: CGFloat = 0.32

    case extended(tip: CGPoint, height: CGFloat, spread: CGFloat, thumbDistance: CGFloat?, allowsClick: Bool = true)
    case pinching(thumbDistance: CGFloat)
    case other
    case lost

    static func classify(index: Finger, middle: Finger, thumb: CGPoint?, palmWidth: CGFloat,
                         allowsClick: Bool = true) -> HandGesture {
        guard palmWidth >= 0.025 else { return .lost }
        let extended = (index.straightness ?? 0) > 0.8 && (middle.straightness ?? 0) > 0.8
            && index.reach > 0.55 * palmWidth && middle.reach > 0.55 * palmWidth
        let spread = hypot(index.tip.x - middle.tip.x, index.tip.y - middle.tip.y) / palmWidth
        let thumbDistance = thumb.map {
            min(hypot($0.x - index.tip.x, $0.y - index.tip.y),
                hypot($0.x - middle.tip.x, $0.y - middle.tip.y)) / palmWidth
        }
        guard extended else {
            // Allow the fingers to bend naturally as the thumb reaches their tips.
            if allowsClick, spread <= 0.48, let thumbDistance, thumbDistance <= clickDistance,
               index.reach > 0.45 * palmWidth, middle.reach > 0.45 * palmWidth {
                return .pinching(thumbDistance: thumbDistance)
            }
            return .other
        }
        return .extended(tip: CGPoint(x: (index.tip.x + middle.tip.x) / 2,
                                      y: (index.tip.y + middle.tip.y) / 2),
                         height: (index.base.y + middle.base.y) / 2,
                         spread: spread, thumbDistance: thumbDistance, allowsClick: allowsClick)
    }
}

private final class HandFrameTracker: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    struct Progress {
        var lastSample: TimeInterval?
        var lastSuccess: TimeInterval?
        var lastError: Int?
        var lastFreshResult: TimeInterval?
        var staleSince: TimeInterval?
        var stableSince: TimeInterval?
    }

    struct Timings {
        private static let bucketWidth: TimeInterval = 0.004
        private var buckets = [Int](repeating: 0, count: 257)
        private(set) var count = 0

        mutating func record(_ seconds: TimeInterval) {
            guard seconds.isFinite, seconds >= 0 else { return }
            let bucket = Int(min(seconds / Self.bucketWidth, Double(buckets.count - 1)))
            buckets[bucket] += 1
            count += 1
        }

        var p95Milliseconds: String {
            guard count > 0 else { return "n/a" }
            let rank = count - count / 20
            var cumulative = 0
            for (index, quantity) in buckets.enumerated() {
                cumulative += quantity
                if cumulative >= rank {
                    return index == buckets.count - 1 ? ">=1024" : String((index + 1) * 4)
                }
            }
            return "n/a"
        }
    }

    struct Metrics {
        var delivered = 0
        var overwritten = 0
        var captureDrops = 0
        var usablePoses = 0
        var staleResults = 0
        var visionErrors = 0
        var captureAge = Timings()
        var vision = Timings()
        var resultAge = Timings()

        func summary(over duration: TimeInterval) -> String {
            let fps = String(format: "%.1f", Double(delivered) / duration)
            return "delivered=\(delivered) fps=\(fps) results=\(resultAge.count) usable=\(usablePoses) stale=\(staleResults) errors=\(visionErrors) overwritten=\(overwritten) avDrops=\(captureDrops) captureSamples=\(captureAge.count) p95_ms capture=\(captureAge.p95Milliseconds) vision=\(vision.p95Milliseconds) result=\(resultAge.p95Milliseconds)"
        }
    }

    private struct Frame: @unchecked Sendable {
        let buffer: CVPixelBuffer
        let timestamp: TimeInterval
    }

    private struct State {
        var stopped = false
        var processing = false
        var pending: Frame?
        var progress = Progress()
        var captureClock: CMClock?
        var metrics = Metrics()
    }

    private let state = OSAllocatedUnfairLock(initialState: State())
    private let request = VNDetectHumanHandPoseRequest()
    private let visionQueue = DispatchQueue(label: "baijiu.hand-cursor.vision", qos: .userInteractive)
    private let callbackQueue: DispatchQueue
    private let onGesture: (HandGesture, TimeInterval, Bool) -> Void

    init(callbackQueue: DispatchQueue, onGesture: @escaping (HandGesture, TimeInterval, Bool) -> Void) {
        self.callbackQueue = callbackQueue
        self.onGesture = onGesture
        super.init()
        request.maximumHandCount = 2
    }

    var progress: Progress { state.withLock { $0.progress } }

    func setCaptureClock(_ clock: CMClock?) { state.withLock { $0.captureClock = clock } }

    func takeMetrics() -> Metrics {
        state.withLock {
            let metrics = $0.metrics
            $0.metrics = Metrics()
            return metrics
        }
    }

    func stop() {
        state.withLock {
            $0.stopped = true
            $0.pending = nil
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let frame = Frame(buffer: buffer, timestamp: ProcessInfo.processInfo.systemUptime)
        // The capture PTS and this clock share a timebase. Keep gesture timestamps in uptime.
        let captureAge = state.withLock { $0.captureClock }.map {
            CMTimeSubtract(CMClockGetTime($0), sampleBuffer.presentationTimeStamp).seconds
        }
        let shouldStart = state.withLock { state in
            guard !state.stopped else { return false }
            state.progress.lastSample = frame.timestamp
            state.metrics.delivered += 1
            if let captureAge { state.metrics.captureAge.record(captureAge) }
            if state.processing {
                if state.pending != nil { state.metrics.overwritten += 1 }
                state.pending = frame
                return false
            }
            state.processing = true
            return true
        }
        if shouldStart { process(frame) }
    }

    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        state.withLock { $0.metrics.captureDrops += 1 }
    }

    private func process(_ frame: Frame) {
        // Keep the newest pending frame and let AVFoundation tear down even if Vision is blocked.
        visionQueue.async { [self] in
            guard state.withLock({ !$0.stopped }) else { return }
            let gesture: HandGesture
            let errorCode: Int?
            let startedAt = ProcessInfo.processInfo.systemUptime
            let handler = VNImageRequestHandler(cvPixelBuffer: frame.buffer, orientation: .up)
            do {
                try handler.perform([request])
                gesture = HandCursor.gesture(in: request.results ?? [])
                errorCode = nil
            } catch {
                gesture = .lost
                errorCode = (error as NSError).code
            }
            let completedAt = ProcessInfo.processInfo.systemUptime
            guard state.withLock({ state in
                guard !state.stopped else { return false }
                state.metrics.vision.record(completedAt - startedAt)
                state.progress.lastError = errorCode
                if errorCode == nil {
                    state.progress.lastSuccess = completedAt
                }
                return true
            }) else { return }
            let timestamp = frame.timestamp
            callbackQueue.async { [self] in
                recordResult(gesture, timestamp: timestamp, succeeded: errorCode == nil)
                onGesture(gesture, timestamp, errorCode == nil)
                let next = state.withLock { state in
                    let pending = state.pending
                    state.pending = nil
                    state.processing = pending != nil
                    return pending
                }
                if let next { process(next) }
            }
        }
    }

    private func recordResult(_ gesture: HandGesture, timestamp: TimeInterval, succeeded: Bool) {
        let now = ProcessInfo.processInfo.systemUptime
        let age = now - timestamp
        state.withLock { state in
            guard !state.stopped else { return }
            state.metrics.resultAge.record(age)
            guard succeeded else { state.metrics.visionErrors += 1; return }
            switch gesture {
            case .extended, .pinching: state.metrics.usablePoses += 1
            case .other, .lost: break
            }
            if age >= HandCursor.maximumResultAge {
                state.metrics.staleResults += 1
                if state.progress.staleSince == nil { state.progress.staleSince = now }
                state.progress.stableSince = nil
            } else {
                state.progress.staleSince = nil
                if state.progress.stableSince == nil || now - (state.progress.lastFreshResult ?? -.infinity) >= 1 {
                    state.progress.stableSince = now
                }
                state.progress.lastFreshResult = now
            }
        }
    }
}

// Each activation owns its pointer. Stop synchronizes with an in-flight event
// so callbacks from that activation cannot move the cursor afterward.
final class HandPointerOutput: @unchecked Sendable {
    private let pointer = OSAllocatedUnfairLock<HandPointer?>(initialState: nil)

    func start() { pointer.withLock { $0 = HandPointer() } }

    func stop() { pointer.withLock { $0 = nil } }

    func receive(_ gesture: HandGesture, _ timestamp: TimeInterval) {
        pointer.withLock { state in
            guard state != nil else { return }
            let age = ProcessInfo.processInfo.systemUptime - timestamp
            guard age < HandCursor.maximumResultAge else {
                state = HandPointer()
                return
            }
            let bounds = CGDisplayBounds(CGMainDisplayID())
            guard let event = state?.event(for: gesture, at: timestamp, in: bounds) else { return }
            if case .click = event, age >= 0.2 {
                state = HandPointer()
                return
            }
            event.post()
        }
    }
}

struct HandPointer {
    private static let movementSensitivity: CGFloat = 1.5
    private static let scrollPixelsPerFrameHeight: CGFloat = 9000
    private static let scrollNoiseRadius: CGFloat = 0.001
    private static let spreadOpen: CGFloat = 0.51
    private static let spreadClosed: CGFloat = 0.44
    private static let scrollStartDistance: CGFloat = 0.004
    private static let maxReacquisitionDelta: CGFloat = 0.045
    private static let spreadClickPulse: ClosedRange<TimeInterval> = 0.03...0.28
    private static let spreadClickWindow: TimeInterval = 0.72

    private enum SpreadClick {
        case ready
        case open(origin: CGPoint, startedAt: TimeInterval, openedAt: TimeInterval, second: Bool, height: CGFloat)
        case joinedOnce(origin: CGPoint, startedAt: TimeInterval, joinedAt: TimeInterval, height: CGFloat)
    }

    enum Event {
        case move(CGPoint)
        case click(CGPoint)
        case scroll(Int32)

        func post() {
            switch self {
            case .move(let point):
                CGEvent(mouseEventSource: nil, mouseType: .mouseMoved,
                        mouseCursorPosition: point, mouseButton: .left)?.post(tap: .cghidEventTap)
            case .click(let point):
                guard let down = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown,
                                         mouseCursorPosition: point, mouseButton: .left),
                      let up = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp,
                                       mouseCursorPosition: point, mouseButton: .left) else { return }
                down.setIntegerValueField(.mouseEventClickState, value: 1)
                up.setIntegerValueField(.mouseEventClickState, value: 1)
                Event.move(point).post()
                down.post(tap: .cghidEventTap)
                up.post(tap: .cghidEventTap)
            case .scroll(let pixels):
                CGEvent(scrollWheelEvent2Source: nil, units: .pixel, wheelCount: 1,
                        wheel1: pixels, wheel2: 0, wheel3: 0)?.post(tap: .cghidEventTap)
            }
        }
    }

    private var displayBounds: CGRect?
    private var filtered: CGPoint?
    private var posted: CGPoint?
    private var lastSampleAt: TimeInterval?
    private var clickArmed = false
    private var lastPointingAt: TimeInterval?
    private var thumbNearSince: TimeInterval?
    private var scrollHeight: CGFloat?
    private var lastPalmHeight: CGFloat?
    private var uncertainSince: TimeInterval?
    private var spreadClick: SpreadClick?

    mutating func event(for gesture: HandGesture, at time: TimeInterval, in bounds: CGRect) -> Event? {
        let interval = lastSampleAt.map { time - $0 } ?? 0
        if displayBounds != bounds || interval >= 0.5 {
            self = HandPointer()
            displayBounds = bounds
        } else if interval > 0.18 {
            // Slow frames may still move or scroll, but cannot complete an old click.
            spreadClick = nil
            lastPointingAt = nil
            clickArmed = false
            thumbNearSince = nil
        }
        lastSampleAt = time
        if let lastPointingAt, time - lastPointingAt > 0.25 {
            self.lastPointingAt = nil
            clickArmed = false
            thumbNearSince = nil
        }
        if let uncertainSince, time - uncertainSince > 0.12 {
            scrollHeight = nil
            lastPalmHeight = nil
        }
        switch gesture {
        case .extended(let tip, let height, let spread, let thumbDistance, let allowsClick):
            if !allowsClick {
                spreadClick = nil
                clickArmed = false
                thumbNearSince = nil
            }
            if let lastPalmHeight, (uncertainSince != nil || interval > 0.18),
               abs(height - lastPalmHeight) > Self.maxReacquisitionDelta {
                // Rebase after an unobserved jump rather than sending it as one large wheel event.
                self.lastPalmHeight = nil
                if scrollHeight != nil { scrollHeight = height }
            }
            uncertainSince = nil
            let previousHeight = lastPalmHeight ?? height
            lastPalmHeight = height
            if spread >= Self.spreadOpen || (scrollHeight != nil && spread > Self.spreadClosed) {
                lastPointingAt = nil
                clickArmed = false
                thumbNearSince = nil
                if scrollHeight == nil {
                    scrollHeight = previousHeight
                    openSpread(at: time, height: previousHeight)
                }
                return scroll(to: height, previousHeight: previousHeight, at: time)
            }

            let movementInterval = lastPointingAt.map { time - $0 } ?? interval
            lastPointingAt = time
            // A stroke may end on the same frame on which the fingers come together.
            var finalScroll: Event?
            if case .open = spreadClick, let anchor = scrollHeight,
               Self.isScrollStroke(to: height, from: anchor, previousHeight: previousHeight) {
                spreadClick = nil
                finalScroll = scroll(to: height, previousHeight: previousHeight, at: time)
            }
            scrollHeight = nil
            let origin = allowsClick ? closeSpread(at: time, height: height) : nil
            let thumbClicked = allowsClick && shouldClick(thumbDistance: thumbDistance, at: time)
            if let finalScroll { return finalScroll }
            if let origin {
                posted = origin
                filtered = origin
                return .click(origin)
            }
            let movement = position(for: tip, after: movementInterval, in: bounds)
            if thumbClicked, let posted {
                spreadClick = .ready
                return .click(posted)
            }
            return movement.map(Event.move)
        case .pinching(let thumbDistance):
            scrollHeight = nil
            lastPalmHeight = nil
            uncertainSince = nil
            spreadClick = nil
            guard lastPointingAt != nil else { return nil }
            if shouldClick(thumbDistance: thumbDistance, at: time), let posted {
                return .click(posted)
            }
            return nil
        case .other:
            thumbNearSince = nil
            if uncertainSince == nil { uncertainSince = time }
            spreadClick = nil
            return nil
        case .lost:
            self = HandPointer()
            return nil
        }
    }

    private static func isScrollStroke(to height: CGFloat, from anchor: CGFloat, previousHeight: CGFloat) -> Bool {
        let distance = abs(height - anchor)
        let step = height - previousHeight
        let offset = previousHeight - anchor
        return distance >= scrollStartDistance
            || (distance >= 0.0025 && abs(step) >= 0.0008 && abs(offset) >= 0.0008 && step * offset > 0)
    }

    private mutating func scroll(to height: CGFloat, previousHeight: CGFloat, at time: TimeInterval) -> Event? {
        guard let anchor = scrollHeight else { return nil }
        if case .open(_, let startedAt, let openedAt, _, _) = spreadClick {
            if time - openedAt <= Self.spreadClickPulse.upperBound,
               time - startedAt <= Self.spreadClickWindow,
               !Self.isScrollStroke(to: height, from: anchor, previousHeight: previousHeight) {
                return nil
            }
            spreadClick = nil
        }
        let delta = height - anchor
        let distance = max(0, abs(delta) - Self.scrollNoiseRadius)
        let pixels = Int32(distance * Self.scrollPixelsPerFrameHeight) * (delta < 0 ? -1 : 1)
        guard pixels != 0 else { return nil }
        // Carry the remainder forward so slow strokes don't lose distance on every frame.
        scrollHeight = anchor + CGFloat(pixels) / Self.scrollPixelsPerFrameHeight
        return .scroll(pixels)
    }

    private mutating func openSpread(at time: TimeInterval, height: CGFloat) {
        switch spreadClick {
        case .joinedOnce(let origin, let startedAt, let joinedAt, let firstHeight)
            where time - startedAt <= Self.spreadClickWindow
                && time - joinedAt >= Self.spreadClickPulse.lowerBound
                && abs(height - firstHeight) < Self.scrollStartDistance:
            spreadClick = .open(origin: origin, startedAt: startedAt, openedAt: time, second: true, height: firstHeight)
        case .ready, .joinedOnce:
            spreadClick = posted.map { .open(origin: $0, startedAt: time, openedAt: time, second: false, height: height) }
        default:
            spreadClick = nil
        }
    }

    private mutating func closeSpread(at time: TimeInterval, height: CGFloat) -> CGPoint? {
        switch spreadClick {
        case .open(let origin, let startedAt, let openedAt, let second, let firstHeight)
            where Self.spreadClickPulse.contains(time - openedAt)
                && time - startedAt <= Self.spreadClickWindow
                && abs(height - firstHeight) < Self.scrollStartDistance:
            spreadClick = second ? .ready : .joinedOnce(origin: origin, startedAt: startedAt,
                                                       joinedAt: time, height: firstHeight)
            return second ? origin : nil
        case .joinedOnce(_, let startedAt, _, let firstHeight)
            where time - startedAt <= Self.spreadClickWindow && abs(height - firstHeight) < Self.scrollStartDistance:
            return nil
        default:
            spreadClick = .ready
            return nil
        }
    }

    private mutating func shouldClick(thumbDistance: CGFloat?, at time: TimeInterval) -> Bool {
        if let thumbDistance, thumbDistance >= 0.46 { clickArmed = true }
        guard clickArmed, let thumbDistance, thumbDistance <= HandGesture.clickDistance else {
            thumbNearSince = nil
            return false
        }
        if thumbNearSince == nil { thumbNearSince = time }
        guard let thumbNearSince, time - thumbNearSince >= 0.025 else { return false }
        clickArmed = false
        self.thumbNearSince = nil
        return true
    }

    private mutating func position(for tip: CGPoint, after interval: TimeInterval, in bounds: CGRect) -> CGPoint? {
        // Mirror camera x for selfie-style motion and convert Vision's upward y to screen y.
        let x = min(1, max(0, 0.5 + (0.5 - tip.x) / 0.64 * Self.movementSensitivity))
        let y = min(1, max(0, 0.5 + (0.5 - tip.y) / 0.64 * Self.movementSensitivity))
        let target = CGPoint(x: bounds.minX + x * max(0, bounds.width - 1),
                             y: bounds.minY + y * max(0, bounds.height - 1))
        let previous = filtered ?? target
        let dx = target.x - previous.x
        let dy = target.y - previous.y
        let screenDistance = hypot(dx, dy) / max(1, min(bounds.width, bounds.height))
        // Respond to deliberate movement while smoothing near-stationary samples.
        let motion = min(1, max(0, (screenDistance - 0.003) / 0.011))
        let timeConstant = 0.085 - motion * (0.085 - 0.010)
        let gain = 1 - exp(-max(0, interval) / timeConstant)
        let point = CGPoint(x: previous.x + gain * dx, y: previous.y + gain * dy)
        filtered = point
        if let posted, hypot(point.x - posted.x, point.y - posted.y) < 1 { return nil }
        posted = point
        return point
    }
}
