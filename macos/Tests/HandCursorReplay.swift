import Foundation
import Vision

// Run from the project root:
// swiftc -parse-as-library Sources/HandCursor.swift Tests/HandCursorReplay.swift -o /tmp/baijiu-hand-cursor-replay
// /tmp/baijiu-hand-cursor-replay
@main
struct HandCursorReplay {
    static func main() {
        var failures = 0
        func check(_ name: String, _ frames: [HandGesture], interval: TimeInterval = 1.0 / 30,
                   passes: ([HandPointer.Event?]) -> Bool) {
            let bounds = CGRect(x: 0, y: 0, width: 1200, height: 800)
            var pointer = HandPointer()
            let events = frames.enumerated().map { index, gesture in
                pointer.event(for: gesture, at: Double(index) * interval, in: bounds)
            }
            if passes(events) {
                print("PASS \(name)")
            } else {
                failures += 1
                print("FAIL \(name)")
                for (index, event) in events.enumerated() {
                    if let event { print("  frame \(index): \(event)") }
                }
            }
        }

        check("two short vertical strokes scroll without clicking", [
            pose(), pose(spread: 0.70),
            pose(spread: 0.70, height: 0.405), pose(spread: 0.70, height: 0.405),
            pose(height: 0.405), pose(height: 0.405),
            pose(spread: 0.70, height: 0.405),
            pose(spread: 0.70, height: 0.410), pose(spread: 0.70, height: 0.410),
            pose(height: 0.410)
        ]) { events in
            [2, 7].allSatisfy { (scroll(events[$0]) ?? 0) > 0 }
                && events.compactMap(click).isEmpty
        }

        check("vertical motion scrolls on the frame the fingers spread", [
            pose(), pose(spread: 0.70, height: 0.416),
            pose(spread: 0.70, height: 0.416), pose(height: 0.416)
        ]) { events in
            (scroll(events[1]) ?? 0) > 0 && events.compactMap(click).isEmpty
        }

        check("spread measurements near the threshold do not click", [
            pose(), pose(spread: 0.50), pose(spread: 0.50),
            pose(spread: 0.46), pose(spread: 0.46),
            pose(spread: 0.50), pose(spread: 0.50), pose(spread: 0.46)
        ]) { events in
            events.compactMap(click).isEmpty && events.compactMap(scroll).isEmpty
        }

        check("scrolling continues when spread measurements fluctuate", [
            pose(), pose(spread: 0.70),
            pose(spread: 0.46, height: 0.405), pose(spread: 0.50, height: 0.410),
            pose(spread: 0.46, height: 0.415), pose(spread: 0.50, height: 0.420),
            pose(spread: 0.46, height: 0.425), pose(spread: 0.50, height: 0.430),
            pose(height: 0.430)
        ]) { events in
            events[2...7].allSatisfy { (scroll($0) ?? 0) > 0 }
                && events.compactMap(click).isEmpty
        }

        let smallDownwardSteps = (1...30).map { step in
            pose(spread: 0.70, height: 0.40 - CGFloat(step) * 0.001)
        }
        check("small downward steps keep scrolling across a full second",
              [pose(), pose(spread: 0.70)] + smallDownwardSteps) { events in
            [2...11, 12...21, 22...31].allSatisfy { frames in
                events[frames].contains { (scroll($0) ?? 0) < 0 }
            } && events.compactMap(scroll).allSatisfy { $0 < 0 }
                && events.compactMap(click).isEmpty
        }

        check("scrolling resumes on the first clear sample after one uncertain frame", [
            pose(), pose(spread: 0.70), pose(spread: 0.70, height: 0.410),
            .other, pose(spread: 0.70, height: 0.430), pose(spread: 0.70, height: 0.440),
            pose(height: 0.440)
        ]) { events in
            [2, 4, 5].allSatisfy { (scroll(events[$0]) ?? 0) > 0 }
                && events[3] == nil && events.compactMap(click).isEmpty
        }

        check("slow frame delivery still scrolls", [
            pose(), pose(spread: 0.70), pose(spread: 0.70, height: 0.41),
            pose(spread: 0.70, height: 0.42), pose(height: 0.42)
        ], interval: 0.42) { events in
            events.compactMap(scroll).count == 2 && events.compactMap(click).isEmpty
        }

        check("pointing catches up after a single uncertain sample", [
            pose(), .other, pose(x: 0.498)
        ]) { events in
            guard let first = move(events[0]), let next = move(events[2]) else { return false }
            // The physical step maps to 5.6 points. Catch up by at least half of it.
            return next.x - first.x >= 2.8
        }

        check("a hidden pinky does not freeze pointing or arm a click", [
            pose(), partialPose(x: 0.49), partialPose(x: 0.48)
        ]) { events in
            events[1...2].allSatisfy { move($0) != nil }
                && events.compactMap(click).isEmpty && events.compactMap(scroll).isEmpty
        }

        check("a large jump between slow frames rebases scrolling", [
            pose(), pose(spread: 0.70), pose(spread: 0.70, height: 0.41),
            pose(spread: 0.70, height: 0.65), pose(spread: 0.70, height: 0.66)
        ], interval: 0.22) { events in
            events[3] == nil && (scroll(events[4]) ?? 0) > 0 && events.compactMap(click).isEmpty
        }

        check("a long tracking gap does not replay old scroll motion",
              [pose(), pose(spread: 0.70), pose(spread: 0.70, height: 0.41)]
              + Array(repeating: .other, count: 6)
              + [pose(spread: 0.70, height: 0.52), pose(spread: 0.70, height: 0.53)]) { events in
            events[9] == nil && (scroll(events[10]) ?? 0) > 0 && events.compactMap(click).isEmpty
        }

        let doubleSpread = [
            pose(), pose(spread: 0.70), pose(spread: 0.70),
            pose(x: 0.49), pose(x: 0.49),
            pose(spread: 0.70, x: 0.49), pose(spread: 0.70, x: 0.49), pose(x: 0.49)
        ]
        check("two stationary spreads click once at the original pointer", doubleSpread) { events in
            guard let origin = move(events[0]) else { return false }
            return events.compactMap(click) == [origin] && click(events[7]) == origin
                && events.compactMap(scroll).isEmpty
        }

        check("stationary spreads at different palm heights do not click", [
            pose(), pose(spread: 0.70), pose(spread: 0.70), pose(),
            pose(height: 0.43), pose(height: 0.45), pose(height: 0.45),
            pose(spread: 0.70, height: 0.45), pose(spread: 0.70, height: 0.45), pose(height: 0.45)
        ]) { events in
            events.compactMap(click).isEmpty && events.compactMap(scroll).isEmpty
        }

        check("a thumb click can immediately follow a spread click", doubleSpread + [
            pose(x: 0.49, thumbDistance: 0.20), pose(x: 0.49, thumbDistance: 0.20)
        ]) { events in
            events.compactMap(click).count == 2 && click(events[7]) != nil && click(events[9]) != nil
        }

        check("a lost hand clears the old spread and allows a fresh click", [
            pose(), pose(spread: 0.70), pose(spread: 0.70), pose(), pose(), .lost,
            pose(x: 0.40), pose(spread: 0.70, x: 0.40), pose(spread: 0.70, x: 0.40),
            pose(x: 0.40), pose(x: 0.40),
            pose(spread: 0.70, x: 0.40), pose(spread: 0.70, x: 0.40), pose(x: 0.40)
        ]) { events in
            guard let reacquired = move(events[6]) else { return false }
            return events.compactMap(click) == [reacquired] && click(events[13]) == reacquired
        }

        let (firstMinute, lastMinute) = sustainedReplay()
        if firstMinute.clicks == 60 && firstMinute.clicks == lastMinute.clicks
            && firstMinute.scrolls > 0 && firstMinute.scrolls == lastMinute.scrolls
            && firstMinute.scrollPixels > 0 && firstMinute.scrollPixels == lastMinute.scrollPixels
            && firstMinute.moves > 0 && abs(firstMinute.moves - lastMinute.moves) <= 2 {
            print("PASS pointing, scrolling, and clicking stay consistent over 20 minutes")
        } else {
            failures += 1
            print("FAIL pointing, scrolling, and clicking stay consistent over 20 minutes")
            print("  first minute: \(firstMinute)")
            print("  last minute: \(lastMinute)")
        }

        if failures > 0 { exit(1) }
    }

    private struct EventTotals {
        var moves = 0
        var scrolls = 0
        var scrollPixels = 0
        var clicks = 0
    }

    private static func sustainedReplay() -> (EventTotals, EventTotals) {
        let pointing = (0...14).map { pose(x: 0.50 - CGFloat($0) * 0.0015) }
        let scrolling: [HandGesture] = (0...9).map { frame in
            if frame == 3 { return .other }
            return pose(spread: 0.70, height: 0.40 + CGFloat(frame) * 0.004, x: 0.48)
        }
        let joined = pose(height: 0.436, x: 0.48)
        let spread = pose(spread: 0.70, height: 0.436, x: 0.48)
        let thumbNear = pose(height: 0.436, x: 0.48, thumbDistance: 0.20)
        let cycle = pointing + [pose(x: 0.48)] + scrolling + Array(repeating: joined, count: 5)
            + [spread, spread, joined, joined, spread, spread, joined, thumbNear, thumbNear]
            + Array(repeating: pose(), count: 20)
        let framesPerSecond = 30
        let framesPerMinute = 60 * framesPerSecond
        let bounds = CGRect(x: 0, y: 0, width: 1440, height: 900)
        var pointer = HandPointer()
        var totals = [EventTotals(), EventTotals()]
        for frame in 0..<(20 * framesPerMinute) {
            let event = pointer.event(for: cycle[frame % cycle.count],
                                      at: Double(frame) / Double(framesPerSecond), in: bounds)
            let minute = frame / framesPerMinute
            guard minute == 0 || minute == 19 else { continue }
            let window = minute == 0 ? 0 : 1
            switch event {
            case .move: totals[window].moves += 1
            case .scroll(let pixels):
                totals[window].scrolls += 1
                totals[window].scrollPixels += Int(pixels)
            case .click: totals[window].clicks += 1
            case nil: break
            }
        }
        return (totals[0], totals[1])
    }

    private static func pose(spread: CGFloat = 0.15, height: CGFloat = 0.40, x: CGFloat = 0.50,
                             thumbDistance: CGFloat = 0.80) -> HandGesture {
        let palmWidth: CGFloat = 0.15
        let halfSpread = palmWidth * spread / 2
        let tipY = height + 0.20
        let indexTip = CGPoint(x: x - halfSpread, y: tipY)
        let middleTip = CGPoint(x: x + halfSpread, y: tipY)
        let thumb = CGPoint(x: indexTip.x - thumbDistance * palmWidth, y: tipY)

        func finger(baseX: CGFloat, tip: CGPoint) -> Finger {
            Finger(base: CGPoint(x: baseX, y: height),
                   joint: CGPoint(x: (baseX + tip.x) / 2, y: (height + tip.y) / 2), tip: tip)
        }

        return HandGesture.classify(index: finger(baseX: x - 0.025, tip: indexTip),
                                    middle: finger(baseX: x + 0.025, tip: middleTip),
                                    thumb: thumb, palmWidth: palmWidth)
    }

    private static func partialPose(x: CGFloat) -> HandGesture {
        let points: [VNHumanHandPoseObservation.JointName: CGPoint] = [
            .indexMCP: CGPoint(x: x - 0.025, y: 0.40), .indexPIP: CGPoint(x: x - 0.02, y: 0.50),
            .indexTip: CGPoint(x: x - 0.015, y: 0.60), .middleMCP: CGPoint(x: x + 0.025, y: 0.40),
            .middlePIP: CGPoint(x: x + 0.02, y: 0.50), .middleTip: CGPoint(x: x + 0.015, y: 0.60),
            .ringMCP: CGPoint(x: x + 0.075, y: 0.40), .thumbTip: CGPoint(x: x - 0.015, y: 0.61)
        ]
        return HandCursor.gesture { points[$0] }
    }

    private static func move(_ event: HandPointer.Event?) -> CGPoint? {
        if case .move(let point) = event { return point }
        return nil
    }

    private static func click(_ event: HandPointer.Event?) -> CGPoint? {
        if case .click(let point) = event { return point }
        return nil
    }

    private static func scroll(_ event: HandPointer.Event?) -> Int32? {
        if case .scroll(let pixels) = event { return pixels }
        return nil
    }
}
