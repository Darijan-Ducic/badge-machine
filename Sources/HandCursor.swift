import AVFoundation
import CoreGraphics
import Vision

final class HandCursor: @unchecked Sendable {
    private let queue = DispatchQueue(label: "baijiu.hand-cursor")
    private let frameQueue = DispatchQueue(label: "baijiu.hand-cursor.frames")
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private var tracker: HandFrameTracker?
    private var run: UUID?
    private var sessionObservers: [NSObjectProtocol] = []

    func start(onGesture: @escaping (HandGesture, TimeInterval) -> Void, onStart: @escaping (Bool) -> Void,
               onFailure: @escaping () -> Void) {
        queue.async { [self] in
            guard self.configure() else { onStart(false); return }
            let run = UUID()
            self.run = run
            let tracker = HandFrameTracker(onGesture: onGesture)
            self.tracker = tracker
            self.output.setSampleBufferDelegate(tracker, queue: self.frameQueue)
            self.sessionObservers = [AVCaptureSession.runtimeErrorNotification,
                                     AVCaptureSession.didStopRunningNotification].map { notification in
                NotificationCenter.default.addObserver(forName: notification, object: self.session, queue: nil) { [weak self] _ in
                    guard let self else { return }
                    self.queue.async {
                        guard self.run == run else { return }
                        self.releaseCamera()
                        onFailure()
                    }
                }
            }
            self.session.startRunning()
            let running = self.session.isRunning
            if !running { self.releaseCamera() }
            onStart(running)
        }
    }

    func stop(onStop: @escaping () -> Void) {
        queue.async { [self] in
            self.releaseCamera()
            onStop()
        }
    }

    private func releaseCamera() {
        guard run != nil else { return }
        run = nil
        sessionObservers.forEach { NotificationCenter.default.removeObserver($0) }
        sessionObservers.removeAll()
        output.setSampleBufferDelegate(nil, queue: nil)
        tracker = nil
        if session.isRunning { session.stopRunning() }
        session.beginConfiguration()
        session.removeOutput(output)
        for input in session.inputs { session.removeInput(input) }
        session.commitConfiguration()
    }

    private func configure() -> Bool {
        guard let camera = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: camera) else { return false }
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        guard session.canAddInput(input), session.canAddOutput(output) else { return false }
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
        func point(_ joint: VNHumanHandPoseObservation.JointName) -> CGPoint? {
            guard let value = points[joint], value.confidence >= 0.55 else { return nil }
            return value.location
        }
        guard let indexBase = point(.indexMCP), let palmEdge = point(.littleMCP) else { return .lost }
        let palmWidth = hypot(palmEdge.x - indexBase.x, palmEdge.y - indexBase.y)
        guard let indexJoint = point(.indexPIP), let indexTip = point(.indexTip),
              let middleBase = point(.middleMCP), let middleJoint = point(.middlePIP),
              let middleTip = point(.middleTip) else { return .other }
        return HandGesture.classify(
            index: Finger(base: indexBase, joint: indexJoint, tip: indexTip),
            middle: Finger(base: middleBase, joint: middleJoint, tip: middleTip),
            thumb: point(.thumbTip),
            palmWidth: palmWidth
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

    case pointing(CGPoint, thumbDistance: CGFloat?)
    case pinching(thumbDistance: CGFloat)
    case scrolling(height: CGFloat)
    case other
    case lost

    static func classify(index: Finger, middle: Finger, thumb: CGPoint?, palmWidth: CGFloat) -> HandGesture {
        guard palmWidth >= 0.025 else { return .lost }
        let extended = (index.straightness ?? 0) > 0.8 && (middle.straightness ?? 0) > 0.8
            && index.reach > 0.55 * palmWidth && middle.reach > 0.55 * palmWidth
        let spread = hypot(index.tip.x - middle.tip.x, index.tip.y - middle.tip.y) / palmWidth
        if spread > 0.48 {
            return extended ? .scrolling(height: (index.base.y + middle.base.y) / 2) : .other
        }
        let thumbDistance = thumb.map {
            min(hypot($0.x - index.tip.x, $0.y - index.tip.y),
                hypot($0.x - middle.tip.x, $0.y - middle.tip.y)) / palmWidth
        }
        guard extended else {
            // Allow the fingers to bend naturally as the thumb reaches their tips.
            if let thumbDistance, thumbDistance <= clickDistance,
               index.reach > 0.45 * palmWidth, middle.reach > 0.45 * palmWidth {
                return .pinching(thumbDistance: thumbDistance)
            }
            return .other
        }
        return .pointing(CGPoint(x: (index.tip.x + middle.tip.x) / 2,
                                 y: (index.tip.y + middle.tip.y) / 2), thumbDistance: thumbDistance)
    }
}

private final class HandFrameTracker: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let request = VNDetectHumanHandPoseRequest()
    private let onGesture: (HandGesture, TimeInterval) -> Void

    init(onGesture: @escaping (HandGesture, TimeInterval) -> Void) {
        self.onGesture = onGesture
        super.init()
        request.maximumHandCount = 2
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let timestamp = ProcessInfo.processInfo.systemUptime
        let handler = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up)
        do {
            try handler.perform([request])
            onGesture(HandCursor.gesture(in: request.results ?? []), timestamp)
        } catch {
            onGesture(.lost, timestamp)
        }
    }
}

struct HandPointer {
    private static let movementSensitivity: CGFloat = 1.5
    private static let scrollPixelsPerFrameHeight: CGFloat = 7200
    private static let scrollNoiseRadius: CGFloat = 0.0015
    private static let spreadClickMotionTolerance: CGFloat = 0.006
    private static let spreadClickDriftTolerance: CGFloat = 0.003
    private static let spreadClickPulse: ClosedRange<TimeInterval> = 0.04...0.28
    private static let spreadClickWindow: TimeInterval = 0.72

    private enum SpreadClick {
        case ready
        case open(origin: CGPoint, startedAt: TimeInterval, openedAt: TimeInterval, second: Bool, height: CGFloat)
        case joinedOnce(origin: CGPoint, startedAt: TimeInterval, joinedAt: TimeInterval)
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
    private var spreadClick: SpreadClick?

    mutating func event(for gesture: HandGesture, at time: TimeInterval, in bounds: CGRect) -> Event? {
        let interval = lastSampleAt.map { time - $0 } ?? 0
        if displayBounds != bounds || interval > 0.18 {
            self = HandPointer()
            displayBounds = bounds
        }
        lastSampleAt = time
        if let lastPointingAt, time - lastPointingAt > 0.25 {
            self.lastPointingAt = nil
            clickArmed = false
            thumbNearSince = nil
        }
        switch gesture {
        case .pointing(let tip, let thumbDistance):
            lastPointingAt = time
            scrollHeight = nil
            if let origin = closeSpread(at: time) {
                posted = origin
                filtered = origin
                return .click(origin)
            }
            let movement = position(for: tip, after: interval, in: bounds)
            if shouldClick(thumbDistance: thumbDistance, at: time), let posted {
                spreadClick = .ready
                return .click(posted)
            }
            return movement.map(Event.move)
        case .pinching(let thumbDistance):
            scrollHeight = nil
            spreadClick = nil
            guard lastPointingAt != nil else { return nil }
            if shouldClick(thumbDistance: thumbDistance, at: time), let posted {
                return .click(posted)
            }
            return nil
        case .scrolling(let height):
            lastPointingAt = nil
            clickArmed = false
            thumbNearSince = nil
            guard let previousHeight = scrollHeight else {
                scrollHeight = height
                openSpread(at: time, height: height)
                return nil
            }
            let delta = height - previousHeight
            if case .open(let origin, let startedAt, let openedAt, let second, let lastHeight) = spreadClick {
                // Accept a clear stroke immediately, or a smaller one moving steadily in the same direction.
                let step = height - lastHeight
                let offset = lastHeight - previousHeight
                let steadyMotion = abs(delta) >= Self.spreadClickDriftTolerance
                    && abs(step) >= 0.001 && abs(offset) >= 0.001 && step * offset > 0
                if abs(delta) < Self.spreadClickMotionTolerance,
                   !steadyMotion,
                   time - openedAt <= Self.spreadClickPulse.upperBound,
                   time - startedAt <= Self.spreadClickWindow {
                    spreadClick = .open(origin: origin, startedAt: startedAt, openedAt: openedAt,
                                        second: second, height: height)
                    return nil
                }
                spreadClick = nil
            }
            let distance = max(0, abs(delta) - Self.scrollNoiseRadius)
            let pixels = Int32(distance * Self.scrollPixelsPerFrameHeight) * (delta < 0 ? -1 : 1)
            guard pixels != 0 else { return nil }
            // Keep the unposted fraction for the next frame so slow strokes stay continuous.
            scrollHeight = previousHeight + CGFloat(pixels) / Self.scrollPixelsPerFrameHeight
            return .scroll(pixels)
        case .other:
            thumbNearSince = nil
            scrollHeight = nil
            spreadClick = nil
            return nil
        case .lost:
            self = HandPointer()
            return nil
        }
    }

    private mutating func openSpread(at time: TimeInterval, height: CGFloat) {
        switch spreadClick {
        case .joinedOnce(let origin, let startedAt, let joinedAt)
            where time - startedAt <= Self.spreadClickWindow
                && time - joinedAt >= Self.spreadClickPulse.lowerBound:
            spreadClick = .open(origin: origin, startedAt: startedAt, openedAt: time, second: true, height: height)
        case .ready, .joinedOnce:
            spreadClick = posted.map { .open(origin: $0, startedAt: time, openedAt: time, second: false, height: height) }
        default:
            spreadClick = nil
        }
    }

    private mutating func closeSpread(at time: TimeInterval) -> CGPoint? {
        switch spreadClick {
        case .open(let origin, let startedAt, let openedAt, let second, _)
            where Self.spreadClickPulse.contains(time - openedAt)
                && time - startedAt <= Self.spreadClickWindow:
            spreadClick = second ? .ready : .joinedOnce(origin: origin, startedAt: startedAt, joinedAt: time)
            return second ? origin : nil
        case .joinedOnce(_, let startedAt, _) where time - startedAt <= Self.spreadClickWindow:
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
        let motion = min(1, max(0, (screenDistance - 0.003) / 0.02))
        let timeConstant = 0.085 - motion * (0.085 - 0.014)
        let gain = 1 - exp(-max(0, interval) / timeConstant)
        let point = CGPoint(x: previous.x + gain * dx, y: previous.y + gain * dy)
        filtered = point
        if let posted, hypot(point.x - posted.x, point.y - posted.y) < 1 { return nil }
        posted = point
        return point
    }
}
