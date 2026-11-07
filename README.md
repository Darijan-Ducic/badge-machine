# Baijiu Machine

A macOS menu bar app that turns silent lip movements into text at your cursor. Requires macOS 14 or later and Xcode command line tools.

```sh
./scripts/build.sh
open "build/Baijiu Machine.app"
```

Allow Camera access on first launch. Choose a keyboard shortcut in the app and grant Accessibility access in System Settings so Baijiu Machine can insert text in other apps. A function key or a combination such as Option–Space works if macOS has not reserved it.

Focus a text field in another app. Press the shortcut to start recording, articulate a short phrase silently, then press it again to stop and send the clip. Keep the text field focused until the result arrives. The app stays available in the menu bar when its window is closed.

Add `GEMINI_API_KEY` to the Git-ignored `.env.local` file in the project root, as a line in `NAME=value` format. Reopen the app after changing the file. Exported environment variables take precedence when launching the executable directly. Gemini 3.5 Flash-Lite receives the silent video. If `MODEL_API_KEY` is configured, Meta Muse Spark receives only the draft text for refinement. The app deletes the temporary video after processing.
