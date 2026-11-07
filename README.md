# Baijiu Machine

Un piccolo studio macOS per articolare in silenzio e ottenere testo dalla fotocamera. Il marchio usa il carattere 白.

Richiede macOS 14 o successivo e gli strumenti da riga di comando di Xcode.

```sh
./scripts/build.sh
open "build/Baijiu Machine.app"
```

Apri la finestra dall’icona nella barra dei menu. Premi **Avvia** e consenti l’accesso alla fotocamera. Premi **Ferma** dopo aver articolato una frase breve. La clip si chiude automaticamente dopo otto secondi. La fotocamera si spegne al termine della registrazione. Puoi salvare la clip locale dalla finestra. L’app non acquisisce audio.

Per provare la trascrizione con Gemini, esci dall’app dal menu e avvia il binario da un terminale in cui hai già esportato `GEMINI_API_KEY`.

```sh
"./build/Baijiu Machine.app/Contents/MacOS/Baijiu Machine"
```

Con la chiave configurata, l’app invia la clip senza audio a Google al termine di ogni registrazione. Senza chiave, acquisizione e salvataggio funzionano in locale. L’accuratezza sul labiale silenzioso va misurata con clip reali.
