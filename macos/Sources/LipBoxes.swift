import AVFoundation
import CoreGraphics
import Vision

/// Mouth crop boxes for the lip-reading server's `X-Tracker-Boxes` header: `[t_ms, x, y, w, h]` in the recording's
/// pixels (top-left origin), padded around the outer lips with the browser tracker's framing (the server crops the
/// per-coordinate median of the boxes and clamps it to the frame).
enum LipBoxes {
    static let cropPad = 1.6 // crop width = outer-lips width × this (static/tracker.js CROP_PAD)
    static let cropAspect = 0.52 // crop height / width (static/tracker2.js CROP_ASPECT, v1's framing)

    /// The padded crop around outer-lip points in Vision's normalized image space (0…1, lower-left origin), as
    /// [x, y, w, h] in pixels of an image of `size` with a top-left origin. Nil without a measurable mouth width.
    static func crop(lips: [CGPoint], in size: CGSize) -> [Double]? {
        guard lips.count >= 2 else { return nil }
        let (imageWidth, imageHeight) = (Double(size.width), Double(size.height))
        let xs = lips.map { Double($0.x) * imageWidth }
        let ys = lips.map { (1 - Double($0.y)) * imageHeight }
        let (left, right, top, bottom) = (xs.min()!, xs.max()!, ys.min()!, ys.max()!)
        guard right > left else { return nil }
        let width: Double = cropPad * (right - left)
        let height: Double = max(cropAspect * width, cropPad * (bottom - top)) // an open mouth outgrows the usual shape
        let centerX: Double = (left + right) / 2, centerY: Double = (top + bottom) / 2
        return [centerX - width / 2, centerY - height / 2, width, height]
    }

    /// Landmark points (normalized to the face's bounding box) in normalized image space, both lower-left origin.
    static func imagePoints(_ points: [CGPoint], face: CGRect) -> [CGPoint] {
        points.map { CGPoint(x: face.minX + $0.x * face.width, y: face.minY + $0.y * face.height) }
    }

    /// One box per `stride`-th decoded frame of `movie` where Vision finds outer lips, t_ms from the first frame
    /// (the server only uses their median, so every 2nd frame keeps the scan at ~1/3 of the clip's length).
    /// With several faces in view the largest one is the speaker.
    static func scan(_ movie: URL, stride: Int = 2) async throws -> [[Int]] {
        let asset = AVURLAsset(url: movie)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw ScanError.noVideo }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? ScanError.noVideo }

        var boxes: [[Int]] = []
        var start: CMTime?
        var index = -1
        while let sample = output.copyNextSampleBuffer() {
            guard let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
            let time = CMSampleBufferGetPresentationTimeStamp(sample)
            if start == nil { start = time }
            index += 1
            guard index % stride == 0 else { continue }
            let request = VNDetectFaceLandmarksRequest()
            try VNImageRequestHandler(cvPixelBuffer: pixels, options: [:]).perform([request])
            let size = CGSize(width: CVPixelBufferGetWidth(pixels), height: CVPixelBufferGetHeight(pixels))
            guard let face = request.results?.max(by: { area($0.boundingBox) < area($1.boundingBox) }),
                  let lips = face.landmarks?.outerLips?.normalizedPoints,
                  let box = crop(lips: imagePoints(lips, face: face.boundingBox), in: size) else { continue }
            let ms = CMTimeGetSeconds(CMTimeSubtract(time, start!)) * 1000
            boxes.append([Int(ms.rounded())] + box.map { Int($0.rounded()) })
        }
        if reader.status == .failed { throw reader.error ?? ScanError.noVideo }
        return boxes
    }

    private static func area(_ rect: CGRect) -> CGFloat { rect.width * rect.height }

    enum ScanError: LocalizedError {
        case noVideo
        var errorDescription: String? { "The recording has no readable video." }
    }
}
