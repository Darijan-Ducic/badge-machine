# Baijiu Machine

Un piccolo studio macOS per articolare in silenzio e ottenere testo dalla fotocamera. Il marchio usa il carattere 白.

Richiede macOS 14 o successivo e gli strumenti da riga di comando di Xcode.

```sh
./scripts/build.sh
open "build/Baijiu Machine.app"
```

Apri la finestra dall’icona nella barra dei menu. Premi **Avvia** e consenti l’accesso alla fotocamera. Premi **Ferma** dopo aver articolato una frase breve. La clip si chiude automaticamente dopo otto secondi. La fotocamera si spegne al termine della registrazione. Puoi salvare la clip locale dalla finestra. L’app non acquisisce audio.

Per attivare la trascrizione, aggiungi `GEMINI_API_KEY` a `.env.local` nella cartella del progetto. Aggiungi `MODEL_API_KEY` per attivare anche la revisione con Meta Muse Spark. Usa una riga per chiave nel formato `NOME=valore`. Salva e riapri l’app dalla cartella `build`. Git ignora il file.

Puoi anche esportare le chiavi nel terminale da cui avvii il binario. Le chiavi esportate hanno precedenza.

```sh
"./build/Baijiu Machine.app/Contents/MacOS/Baijiu Machine"
```

Al termine della registrazione, Gemini riceve la clip senza audio. Muse Spark riceve solo la bozza di testo, rifinisce la punteggiatura e segnala i passaggi da controllare. Se la revisione fallisce, puoi copiare la bozza di Gemini. Senza la chiave Gemini, acquisizione e salvataggio funzionano in locale. L’accuratezza sul labiale silenzioso va misurata con clip reali.
