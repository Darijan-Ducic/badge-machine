# Baijiu Machine

A macOS menu bar app for silent dictation and camera-based pointer control. Requires macOS 14 or later and Xcode command line tools.

```sh
./scripts/build.sh   # with several signing identities: BAIJIU_CODESIGN_IDENTITY=<hash> ./scripts/build.sh
open "build/Baijiu Machine.app"
```

Allow Camera access on first launch. Choose separate shortcuts for **Silent Whisper** and **Move Cursor**. In either row, click **Choose Shortcut**, press a key or combination, then click **Confirm**. Grant Accessibility access in System Settings so Baijiu Machine can insert text in other apps and move the pointer. A function key or a combination such as Option–Space works if macOS has not reserved it.

To use **Fn / Globe**, press Fn in the shortcut picker and click **Confirm**. If the app shows **Finish setting up Fn**, click **Open Input Monitoring** and enable Baijiu Machine in System Settings. Under **Keyboard**, set **Press Globe key to** to **Do Nothing** to stop macOS from opening Emoji & Symbols. Reopen Baijiu Machine if macOS asks you to. Your chosen key stays saved.

Focus a text field in another app. Press the shortcut to start recording, articulate a short phrase silently, then press it again to stop and send the clip. Keep the text field focused until the result arrives. The app stays available in the menu bar when its window is closed.

Silent Whisper reads lips with the badge-machine server, started from the repository root with `uv run --env-file .env server.py` (it listens on `http://127.0.0.1:8765`). After you stop recording, the app finds your outer lips in every second frame of the clip with Apple Vision and pads each box the way the browser tracker frames its mouth crop (1.6 × the lips' width, 0.52 as tall). It then sends the clip with those boxes (`X-Tracker-Boxes`, in recording pixels) to `POST /vtt/paragraph?user=local`, or to `POST /vtt?user=local` if the server has no paragraph route. The app inserts the lines the server read: each segment's top line from `/vtt/paragraph`, or `lines[0]` from `/vtt`, capitalized and ending with a period. A read takes about 15–25 s. If no lips are found or the server fails, the app shows that in the status and inserts nothing. To use another server, run `defaults write com.darijanducic.baijiumachine serverURL http://host:port` or export `BAIJIU_SERVER_URL`. The app deletes the temporary video after processing.

Check the lip reader from the project root. The unit checks cover box conversion, both response shapes and the `/vtt` fallback. The live check sends one recording to the running server and stores it as an unlabeled clip for user `local`.

```sh
swiftc -parse-as-library Sources/LipBoxes.swift Sources/LipReader.swift Tests/LipReaderTests.swift -o /tmp/baijiu-lip-reader-tests && /tmp/baijiu-lip-reader-tests
swiftc -parse-as-library Sources/LipBoxes.swift Sources/LipReader.swift Tests/LipReaderLive.swift -o /tmp/baijiu-lip-reader-live && /tmp/baijiu-lip-reader-live path/to/take.mov
```

Press the **Move Cursor** shortcut and point with your right index and middle fingers together to move the pointer on the main display. With your hand at the same height, spread and rejoin the two fingers twice quickly to click where the pointer was before the first spread. You can also click by bringing your thumb near the joined fingertips. To scroll, spread the two fingers and move your hand up or down. Bring them together to resume pointing. Lower your hand to pause movement. Press the shortcut again to release the camera. The app processes pointer tracking locally and uses the camera for one mode at a time.
