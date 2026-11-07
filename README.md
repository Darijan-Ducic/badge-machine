# Baijiu Machine

A macOS menu bar app for silent dictation and camera-based pointer control. Requires macOS 14 or later and Xcode command line tools.

```sh
./scripts/build.sh
open "build/Baijiu Machine.app"
```

Allow Camera access on first launch. Choose separate shortcuts for **Silent Whisper** and **Move Cursor**. In either row, click **Choose Shortcut**, press a key or combination, then click **Confirm**. Grant Accessibility access in System Settings so Baijiu Machine can insert text in other apps and move the pointer. A function key or a combination such as Option–Space works if macOS has not reserved it.

To use **Fn / Globe**, allow **Input Monitoring** for Baijiu Machine in System Settings. Under **Keyboard**, set **Press Globe key to** to **Do Nothing** to stop macOS from opening Emoji & Symbols. Reopen Baijiu Machine if macOS asks you to, then press Fn in the shortcut picker and click **Confirm**.

Focus a text field in another app. Press the shortcut to start recording, articulate a short phrase silently, then press it again to stop and send the clip. Keep the text field focused until the result arrives. The app stays available in the menu bar when its window is closed.

Add `GEMINI_API_KEY` to `~/Library/Application Support/Baijiu Machine/.env.local` (or the Git-ignored `.env.local` in the project root for local builds), as a line in `NAME=value` format. Reopen the app after changing the file. Exported environment variables take precedence when launching the executable directly. Gemini 3.5 Flash-Lite receives the silent video. If `MODEL_API_KEY` is configured, Meta Muse Spark receives only the draft text for refinement. The app deletes the temporary video after processing.

Press the **Move Cursor** shortcut and raise your index finger in view of the camera to move the pointer on the main display. Lower your hand to pause movement. Press the shortcut again to release the camera. The app processes pointer tracking locally and uses the camera for one mode at a time.
