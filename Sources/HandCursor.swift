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

    func start(onPoint: @escaping (CGPoint?, TimeInterval) -> Void, onStart: @escaping (Bool) -> Void,
               onFailure: @escaping () -> Void) {
        queue.async { [self] in
            guard self.configure() else { onStart(false); return }
            let run = UUID()
            self.run = run
            let tracker = HandFrameTracker(onPoint: onPoint)
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

    static func rightIndex(in hands: [VNHumanHandPoseObservation]) -> CGPoint? {
        hands.compactMap { hand -> VNRecognizedPoint? in
            guard hand.chirality == .right,
                  let tip = try? hand.recognizedPoint(.indexTip),
                  tip.confidence >= 0.6 else { return nil }
            return tip
        }.max(by: { $0.confidence < $1.confidence })?.location
    }
}

private final class HandFrameTracker: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let request = VNDetectHumanHandPoseRequest()
    private let onPoint: (CGPoint?, TimeInterval) -> Void

    init(onPoint: @escaping (CGPoint?, TimeInterval) -> Void) {
        self.onPoint = onPoint
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
            onPoint(HandCursor.rightIndex(in: request.results ?? []), timestamp)
        } catch {
            onPoint(nil, timestamp)
        }
    }
}

struct HandPointer {
    private var displayBounds: CGRect?
    private var filtered: CGPoint?
    private var posted: CGPoint?

    mutating func position(for tip: CGPoint?, in bounds: CGRect) -> CGPoint? {
        if displayBounds != bounds {
            self = HandPointer()
            displayBounds = bounds
        }
        guard let tip else {
            filtered = nil
            posted = nil
            return nil
        }
        // Mirror camera x for selfie-style motion and convert Vision's upward y to screen y.
        let x = min(1, max(0, (0.82 - tip.x) / 0.64))
        let y = min(1, max(0, (0.82 - tip.y) / 0.64))
        let target = CGPoint(x: bounds.minX + x * max(0, bounds.width - 1),
                             y: bounds.minY + y * max(0, bounds.height - 1))
        let previous = filtered ?? target
        let point = CGPoint(x: previous.x + 0.35 * (target.x - previous.x),
                            y: previous.y + 0.35 * (target.y - previous.y))
        filtered = point
        if let posted, hypot(point.x - posted.x, point.y - posted.y) < 2 { return nil }
        posted = point
        return point
    }
}
