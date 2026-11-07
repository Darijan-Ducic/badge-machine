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
        let palm = Palm(center: CGPoint(x: (indexBase.x + palmEdge.x) / 2,
                                        y: (indexBase.y + palmEdge.y) / 2),
                        width: hypot(palmEdge.x - indexBase.x, palmEdge.y - indexBase.y))
        guard palm.width >= 0.025 else { return .lost }
        // Fingers can briefly hide one another while closing, even while the palm is visible.
        guard let indexJoint = point(.indexPIP), let indexTip = point(.indexTip),
              let middleBase = point(.middleMCP), let middleJoint = point(.middlePIP),
              let middleTip = point(.middleTip) else { return .other }
        return HandGesture.classify(
            index: Finger(base: indexBase, joint: indexJoint, tip: indexTip),
            middle: Finger(base: middleBase, joint: middleJoint, tip: middleTip),
            palm: palm
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

struct Palm {
    let center: CGPoint
    let width: CGFloat

    func isNear(_ other: Palm) -> Bool {
        hypot(center.x - other.center.x, center.y - other.center.y) <= 0.7 * other.width
    }
}

enum HandGesture {
    case pointing(CGPoint, Palm)
    case fist(Palm)
    case other
    case lost

    static func classify(index: Finger, middle: Finger, palm: Palm) -> HandGesture {
        guard palm.width >= 0.025 else { return .lost }
        guard let indexAngle = index.straightness, let middleAngle = middle.straightness else { return .other }
        if indexAngle > 0.8, middleAngle > 0.8,
           index.reach > 0.55 * palm.width, middle.reach > 0.55 * palm.width,
           hypot(index.tip.x - middle.tip.x, index.tip.y - middle.tip.y) <= 0.5 * palm.width {
            return .pointing(CGPoint(x: (index.tip.x + middle.tip.x) / 2,
                                     y: (index.tip.y + middle.tip.y) / 2), palm)
        }
        if indexAngle < 0.3, middleAngle < 0.3,
           index.reach < 0.8 * palm.width, middle.reach < 0.8 * palm.width {
            return .fist(palm)
        }
        return .other
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
    enum Event {
        case move(CGPoint)
        case click(CGPoint)

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
                down.post(tap: .cghidEventTap)
                up.post(tap: .cghidEventTap)
            }
        }
    }

    private var displayBounds: CGRect?
    private var filtered: CGPoint?
    private var posted: CGPoint?
    private var lastSampleAt: TimeInterval?
    private var lastPointingAt: TimeInterval?
    private var armed: (time: TimeInterval, palm: Palm)?
    private var fistSince: TimeInterval?

    mutating func event(for gesture: HandGesture, at time: TimeInterval, in bounds: CGRect) -> Event? {
        let interval = lastSampleAt.map { time - $0 } ?? 0
        if displayBounds != bounds || interval > 0.18 {
            self = HandPointer()
            displayBounds = bounds
        }
        lastSampleAt = time
        switch gesture {
        case .pointing(let tip, let palm):
            fistSince = nil
            // Two recent pointing frames rearm the click, even through a brief uncertain pose.
            if let lastPointingAt, time > lastPointingAt, time - lastPointingAt <= 0.1 {
                armed = (time, palm)
            }
            lastPointingAt = time
            return position(for: tip, after: interval, in: bounds).map(Event.move)
        case .fist(let palm):
            lastPointingAt = nil
            guard let armed, time - armed.time <= 0.65, palm.isNear(armed.palm), let posted else {
                self.armed = nil
                fistSince = nil
                return nil
            }
            if fistSince == nil { fistSince = time }
            guard let fistSince, time - fistSince >= 0.045 else { return nil }
            self.armed = nil
            return .click(posted)
        case .other:
            fistSince = nil
            return nil
        case .lost:
            self = HandPointer()
            return nil
        }
    }

    private mutating func position(for tip: CGPoint, after interval: TimeInterval, in bounds: CGRect) -> CGPoint? {
        // Mirror camera x for selfie-style motion and convert Vision's upward y to screen y.
        let x = min(1, max(0, (0.82 - tip.x) / 0.64))
        let y = min(1, max(0, (0.82 - tip.y) / 0.64))
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
