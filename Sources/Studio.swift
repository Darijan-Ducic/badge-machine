import AppKit
import AVFoundation
import SwiftUI

@MainActor
final class Studio: ObservableObject {
    enum Phase {
        case ready, starting, recording, finishing, transcribing, refining, done, offline, unclear, failed
    }

    @Published private(set) var phase: Phase = .ready
    @Published private(set) var transcript = ""
    @Published private(set) var transcriptNotice: String?
    @Published private(set) var error = ""
    @Published private(set) var clip: URL?
    @Published private(set) var startedAt = Date()
    let camera = Capture()
    private let gemini: Gemini?
    private let muse: MuseSpark?

    init() {
        let keys = APIKeys()
        gemini = keys.gemini.map(Gemini.init(key:))
        muse = keys.meta.map(MuseSpark.init(key:))
    }

    var providerLabel: String {
        guard gemini != nil else { return "REGISTRAZIONE LOCALE" }
        return muse == nil ? "TRASCRIZIONE VIA GEMINI" : "GEMINI + MUSE SPARK"
    }
    var busy: Bool { [.starting, .finishing, .transcribing, .refining].contains(phase) }
    var active: Bool { [.starting, .recording, .finishing].contains(phase) }
    var action: String { phase == .recording ? "Ferma" : "Avvia" }

    var status: String {
        switch phase {
        case .ready: "In attesa"
        case .starting: "Apertura fotocamera"
        case .recording: "Registrazione"
        case .finishing: "Chiusura clip"
        case .transcribing: "Trascrizione in corso"
        case .refining: "Revisione del testo"
        case .done: "Trascrizione pronta"
        case .offline: "Clip pronta"
        case .unclear: "Labiale poco chiaro"
        case .failed: "Serve attenzione"
        }
    }

    func toggle() {
        guard !busy else { return }
        if phase == .recording {
            phase = .finishing
            camera.stop()
        } else {
            transcript = ""
            transcriptNotice = nil
            error = ""
            if let clip { try? FileManager.default.removeItem(at: clip) }
            clip = nil
            phase = .starting
            Task { await start() }
        }
    }

    private func start() async {
        let allowed = await AVCaptureDevice.requestAccess(for: .video)
        guard allowed else {
            error = "Consenti l’accesso in Impostazioni di Sistema → Privacy e sicurezza → Fotocamera."
            phase = .failed
            return
        }
        camera.start(onStart: { [weak self] in
            Task { @MainActor in
                self?.startedAt = Date()
                self?.phase = .recording
            }
        }, onFinish: { [weak self] result in
            Task { @MainActor in await self?.finish(result) }
        })
    }

    private func finish(_ result: Result<URL, Error>) async {
        do {
            let file = try result.get()
            clip = file
            guard let gemini else { phase = .offline; return }
            phase = .transcribing
            guard let draft = try await gemini.transcribe(file) else {
                phase = .unclear
                return
            }
            transcript = draft
            let reviewNotice = "Controlla i passaggi poco chiari prima di copiare."
            if draft.contains(Gemini.uncertaintyMarker) {
                transcriptNotice = reviewNotice
            }
            if let muse {
                phase = .refining
                do {
                    let revision = try await muse.refine(draft)
                    transcript = revision.text
                    if revision.needsReview { transcriptNotice = reviewNotice }
                } catch {
                    transcriptNotice = transcriptNotice == nil
                        ? "Revisione Meta non riuscita. Puoi copiare la bozza di Gemini."
                        : "Revisione Meta non riuscita. \(reviewNotice)"
                }
            }
            phase = .done
        } catch {
            self.error = error.localizedDescription
            phase = .failed
        }
    }

    func copy() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(transcript, forType: .string)
    }

    func saveClip() {
        guard let clip else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "Baijiu Machine.mov"
        panel.allowedContentTypes = [.quickTimeMovie]
        if panel.runModal() == .OK, let destination = panel.url {
            do {
                try Data(contentsOf: clip).write(to: destination, options: .atomic)
            } catch {
                self.error = error.localizedDescription
                phase = .failed
            }
        }
    }

    func discardClip() {
        if let clip { try? FileManager.default.removeItem(at: clip) }
    }
}
