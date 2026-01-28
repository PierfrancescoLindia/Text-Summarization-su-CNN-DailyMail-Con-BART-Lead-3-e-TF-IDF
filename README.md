# Text-Summarization-su-CNN-DailyMail-Con-BART-Lead-3-e-TF-IDF


# Text Summarization su CNN/DailyMail
## Con BART, Lead-3 e TF-IDF (Big Data Analysis)

## 1. Project Overview
Questo progetto realizza un workflow end-to-end di **text summarization** sul benchmark **CNN/DailyMail (v3.0.0)**, confrontando un approccio **abstractive** basato su Transformer con due baseline **extractive** semplici e interpretabili. :contentReference[oaicite:1]{index=1}

L’obiettivo è mettere in evidenza i trade-off tra:
- **capacità di sintesi e riformulazione** (modello neurale fine-tunato);
- **semplicità e fedeltà al testo sorgente** (metodi estrattivi).

I metodi confrontati sono:
- **BART fine-tunato** (facebook/bart-base) per summarization abstractive;
- **Lead-3**, che usa le prime tre frasi dell’articolo come riassunto;
- **TF-IDF + similarità coseno**, che seleziona frasi “rappresentative” dal punto di vista lessicale. :contentReference[oaicite:2]{index=2}

Il progetto è accompagnato da una relazione completa.

**File principali:**
- `Progetto_Big_Data_Analysis_... .ipynb` – notebook con pipeline, esperimenti e risultati
- `Progetto_Big_Data_Analysis_... .pdf` – relazione con metodologia, tabelle e analisi

---

## 2. Data Description
Il dataset **CNN/DailyMail** è composto da articoli giornalistici (campo *article*) e dai relativi riassunti redazionali (campo *highlights*), usati come riferimento (gold summary). :contentReference[oaicite:3]{index=3}

Per rendere l’esperimento sostenibile e riproducibile, il progetto usa subset controllati costruiti con shuffle deterministico (seed fisso):
- **Training subset:** 20.000 esempi (da train)
- **Validation per training:** 1.000 esempi (da validation)
- **Evaluation finale:** 500 esempi (da validation), fissati e riproducibili :contentReference[oaicite:4]{index=4}

---

## 3. Metodologia
### 3.1 Extractive vs Abstractive
- **Extractive:** seleziona frasi dal testo originale (alta fedeltà, spesso interpretabile).
- **Abstractive:** genera un riassunto nuovo riformulando l’informazione (più fluido e compatto, ma più complesso e potenzialmente meno “fedele” in alcuni casi). :contentReference[oaicite:5]{index=5}

### 3.2 Modello Abstractive: BART
BART è un Transformer encoder–decoder pre-addestrato con denoising e successivamente **fine-tunato supervisionatamente** su coppie (articolo, highlights) per specializzarlo nella produzione di riassunti in stile CNN/DailyMail. :contentReference[oaicite:6]{index=6}

### 3.3 Baseline Extractive
- **Lead-3:** sfrutta la “piramide invertita” tipica delle news, concatenando le prime tre frasi.
- **TF-IDF + coseno:** segmenta in frasi, rappresenta con TF-IDF e seleziona le frasi più simili alla rappresentazione complessiva del documento. :contentReference[oaicite:7]{index=7}

### 3.4 Valutazione: ROUGE
Il confronto quantitativo avviene tramite **ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum**, che misurano l’overlap lessicale tra output generato e riassunto di riferimento. :contentReference[oaicite:8]{index=8}  
A questi risultati si affianca una **valutazione qualitativa** su esempi, utile per osservare coerenza, ridondanza e differenze stilistiche non catturate completamente da ROUGE. :contentReference[oaicite:9]{index=9}

---

## 4. Setup Sperimentale (in sintesi)
Gli esperimenti sono stati eseguiti in **Google Colab** con GPU **NVIDIA A100 (40GB)**. :contentReference[oaicite:10]{index=10}  
Per il fine-tuning sono stati adottati vincoli di lunghezza tipici della summarization:
- massimo **512 token** in input (articolo)
- massimo **128 token** in output (riassunto) :contentReference[oaicite:11]{index=11}

Per gestire articoli lunghi e i limiti di input del modello, la pipeline include una strategia di **chunking token-based** e **generazione gerarchica** (riassunti locali → riassunto finale), con controllo dinamico della lunghezza dell’output. :contentReference[oaicite:12]{index=12}

---

## 5. Risultati (ROUGE – Evaluation subset N=500)
I punteggi ROUGE riportati in relazione mostrano un confronto coerente con la natura dei metodi: :contentReference[oaicite:13]{index=13}

- **BART fine-tunato**: migliore su **ROUGE-2** e su misure legate a struttura/coerenza (ROUGE-L / ROUGE-Lsum).
- **Lead-3**: molto competitivo, soprattutto su **ROUGE-1**, tipico nel dominio news.
- **TF-IDF**: prestazioni inferiori, con possibili effetti di frammentarietà/ridondanza. :contentReference[oaicite:14]{index=14}

Valori (come da Tabella in relazione): :contentReference[oaicite:15]{index=15}
- BART: ROUGE-1 0.4164 | ROUGE-2 0.1968 | ROUGE-L 0.2883 | ROUGE-Lsum 0.3850
- Lead-3: ROUGE-1 0.4181 | ROUGE-2 0.1935 | ROUGE-L 0.2649 | ROUGE-Lsum 0.3471
- TF-IDF: ROUGE-1 0.3671 | ROUGE-2 0.1538 | ROUGE-L 0.2419 | ROUGE-Lsum 0.3138

---

## 6. Analisi Qualitativa (sintesi)
L’ispezione manuale degli esempi conferma i trade-off:
- **BART** tende a produrre riassunti più compatti e scorrevoli, con maggiore “sintesi”.
- **Lead-3** è spesso molto informativo e fedele, ma meno conciso e più dipendente dalla struttura dell’articolo.
- **TF-IDF** è interpretabile ma può risultare meno coeso e più ridondante in alcuni casi. :contentReference[oaicite:16]{index=16}

---

## 7. Repository Structure
Struttura consigliata per GitHub:

