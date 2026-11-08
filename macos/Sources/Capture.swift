import AVFoundation
import Foundation

final class Capture: NSObject, AVCaptureFileOutputRecordingDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "baijiu.camera")
    private let output = AVCaptureMovieFileOutput()
    private var started: (() -> Void)?
    private var finished: ((Result<URL, Error>) -> Void)?

    func start(onStart: @escaping () -> Void, onFinish: @escaping (Result<URL, Error>) -> Void) {
        queue.async {
            do {
                if self.session.inputs.isEmpty {
                    guard let camera = AVCaptureDevice.default(for: .video) else { throw CaptureError.unavailable }
                    let input = try AVCaptureDeviceInput(device: camera)
                    guard self.session.canAddInput(input), self.session.canAddOutput(self.output) else {
                        throw CaptureError.unavailable
                    }
                    self.session.beginConfiguration()
                    if self.session.canSetSessionPreset(.vga640x480) {
                        self.session.sessionPreset = .vga640x480
                    }
                    self.session.addInput(input)
                    self.session.addOutput(self.output)
                    self.session.commitConfiguration()
                }
                self.session.startRunning()
                guard self.session.isRunning,
                      let connection = self.output.connection(with: .video),
                      connection.isActive, connection.isEnabled else {
                    if self.session.isRunning { self.session.stopRunning() }
                    throw CaptureError.unavailable
                }
                self.started = onStart
                self.finished = onFinish
                let file = FileManager.default.temporaryDirectory
                    .appendingPathComponent(UUID().uuidString).appendingPathExtension("mov")
                self.output.startRecording(to: file, recordingDelegate: self)
            } catch {
                onFinish(.failure(error))
            }
        }
    }

    func stop() {
        queue.async {
            if self.output.isRecording { self.output.stopRecording() }
        }
    }

    func fileOutput(_ output: AVCaptureFileOutput, didStartRecordingTo fileURL: URL,
                    from connections: [AVCaptureConnection]) {
        queue.async { self.started?() }
    }

    func fileOutput(_ output: AVCaptureFileOutput, didFinishRecordingTo fileURL: URL,
                    from connections: [AVCaptureConnection], error: Error?) {
        queue.async {
            self.session.stopRunning()
            let finish = self.finished
            self.started = nil
            self.finished = nil
            if let error, (error as NSError).userInfo[AVErrorRecordingSuccessfullyFinishedKey] as? Bool != true {
                try? FileManager.default.removeItem(at: fileURL)
                finish?(.failure(error))
            } else {
                finish?(.success(fileURL))
            }
        }
    }

    enum CaptureError: LocalizedError {
        case unavailable
        var errorDescription: String? { "No camera is available." }
    }
}
