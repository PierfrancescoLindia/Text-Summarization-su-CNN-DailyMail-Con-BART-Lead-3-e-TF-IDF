# Text Summarization su CNN/DailyMail Con BART, Lead-3 e TF-IDF

## 1. Project Overview
Questo progetto affronta il problema della **text summarization** su articoli giornalistici, usando il dataset **CNN/DailyMail (v3.0.0)** come benchmark. L’obiettivo è duplice:

1) **Valutare le prestazioni** di un approccio moderno **abstractive** basato su Transformer (BART fine-tunato), confrontandolo con baseline **extractive** semplici ma forti nel dominio news.

2) **Analizzare i trade-off** tra qualità “linguistica” e capacità di sintesi (abstractive) contro semplicità, fedeltà e stabilità (extractive), evidenziando limiti e punti di forza di ciascun metodo.

L’analisi include:
- costruzione di subset riproducibili dal dataset originale;
- implementazione di tre strategie di summarization (BART, Lead-3, TF-IDF);
- valutazione quantitativa con metriche ROUGE e confronto su set fisso;
- valutazione qualitativa su esempi per osservare coerenza, ridondanza e accuratezza informativa. :contentReference[oaicite:1]{index=1}

**File principali:**
- `Progetto_Big_Data_Analysis_... .ipynb` – notebook con pipeline, training e valutazioni
- `Progetto_Big_Data_Analysis_... .pdf` – relazione completa (metodologia + tabelle + analisi) :contentReference[oaicite:2]{index=2}

---

## 2. Data Description
Il dataset **CNN/DailyMail** fornisce coppie (articolo, riassunto di riferimento):
- **article**: testo lungo dell’articolo
- **highlights**: riassunto “gold” redazionale (target supervisionato) :contentReference[oaicite:3]{index=3}

Per rendere l’esperimento sostenibile e riproducibile (vincoli di calcolo e tempo), il progetto utilizza subset ottenuti con shuffle deterministico e seed fisso:
- Training subset: 20.000 esempi
- Validation subset (per controllo training): 1.000 esempi
- Evaluation subset finale: 500 esempi (da validation), fissati e riproducibili :contentReference[oaicite:4]{index=4}

---

## 3. Approcci a confronto: Extractive vs Abstractive
### 3.1 Extractive Summarization
I metodi extractive **non generano nuovo testo**: selezionano frasi dall’articolo originale.
Vantaggi:
- alta fedeltà (minore rischio di “inventare” dettagli);
- semplicità e interpretabilità;
- buone prestazioni in news, dove le informazioni principali sono spesso concentrate all’inizio.
Svantaggi:
- coesione testuale limitata (riassunto può essere “a collage”);
- ridondanza;
- dipendenza dalla struttura dell’articolo.

### 3.2 Abstractive Summarization
I metodi abstractive **generano** un riassunto nuovo, riformulando l’informazione.
Vantaggi:
- migliore capacità di compressione e riorganizzazione;
- riassunti più fluidi e simili a quelli umani.
Svantaggi:
- complessità computazionale;
- possibilità di imprecisioni (hallucination) o omissioni se il contenuto è lungo/complesso;
- sensibilità ai limiti di input (numero massimo di token). :contentReference[oaicite:5]{index=5}

---

## 4. Modello Abstractive: BART e Fine-Tuning
### 4.1 Cos’è BART (perché è adatto alla summarization)
BART è un modello **encoder–decoder** basato su Transformer, progettato come modello di generazione testuale.
L’idea chiave è che:
- **Encoder** legge e “comprende” l’input (l’articolo),
- **Decoder** genera l’output (il riassunto) token per token.

BART nasce con pre-addestramento “denoising”: durante il pretraining impara a ricostruire testo originale partendo da versioni corrotte (es. mascheramenti, permutazioni). Questa strategia lo rende particolarmente efficace in compiti in cui bisogna **trasformare** un testo in un altro testo (come traduzione e summarization). :contentReference[oaicite:6]{index=6}

### 4.2 Impostazione supervisionata per CNN/DailyMail
Nel nostro caso, la summarization è un problema supervisionato:
- input: `article`
- target: `highlights`

Il fine-tuning consiste nel “specializzare” BART (pre-addestrato) per produrre riassunti nello stile e nella distribuzione di CNN/DailyMail.

### 4.3 Preprocessing e vincoli di token
Uno dei problemi principali del dominio news è la lunghezza:
- articoli lunghi spesso superano i limiti di input del modello.

Per questo il progetto lavora con vincoli tipici:
- massimo **512 token** in input
- massimo **128 token** in output :contentReference[oaicite:7]{index=7}

Quando l’articolo eccede il limite, viene applicata una strategia di **chunking token-based** e una generazione **gerarchica**:
1) l’articolo viene diviso in blocchi (chunk) compatibili col limite token,
2) si genera un mini-riassunto per ciascun chunk,
3) si aggregano i mini-riassunti e si genera un riassunto finale,
4) un controllo dinamico della lunghezza mantiene l’output entro soglie coerenti col task. :contentReference[oaicite:8]{index=8}

Questa scelta è importante perché:
- evita di “tagliare” brutalmente il testo perdendo informazioni;
- permette al modello di “vedere” contenuto distribuito lungo tutto l’articolo;
- rende più robusta la pipeline su articoli molto lunghi.

### 4.4 Dettagli del fine-tuning (logica e obiettivo)
Durante il fine-tuning, BART viene ottimizzato per massimizzare la probabilità del riassunto target dato l’articolo.

In termini intuitivi:
- per ogni coppia (article, highlights), il modello viene addestrato a generare il riassunto corretto;
- gli errori nei token generati (rispetto al target) aggiornano i pesi tramite backpropagation;
- il modello apprende sia:
  - quali contenuti selezionare (salienza informativa),
  - come riformularli in modo conciso e coerente.

Il fine-tuning su CNN/DailyMail porta il modello a:
- adottare uno stile giornalistico “riassuntivo” simile agli highlights;
- privilegiare informazioni chiave e ridurre dettagli secondari;
- produrre frasi più scorrevoli rispetto a un collage estrattivo.

### 4.5 Perché BART può superare le baseline
Rispetto a Lead-3 o TF-IDF, BART può:
- comprimere e fondere informazioni provenienti da punti diversi del testo,
- rimuovere ridondanze,
- migliorare coesione e forma linguistica,
- mantenere una struttura da riassunto, non solo una selezione di frasi.

In metrica ROUGE, questo spesso emerge in particolare su **ROUGE-2** (overlap di bigrammi), che tende a premiare riassunti più simili al gold non solo per parole singole ma anche per “frasi”/pattern più corretti. :contentReference[oaicite:9]{index=9}

---

## 5. Baseline Extractive 1: Lead-3
### 5.1 Idea e motivazione
**Lead-3** costruisce il riassunto concatenando le **prime tre frasi** dell’articolo.

Sembra banale, ma nel dominio news è una baseline fortissima perché molti articoli seguono la struttura della **piramide invertita**:
- informazioni più importanti all’inizio,
- dettagli e contesto dopo.

### 5.2 Cosa cattura bene (e perché funziona)
Lead-3 tende a:
- essere molto informativo quando l’articolo è ben scritto “in stile news”;
- mantenere alta fedeltà (frasi originali, nessuna generazione);
- ottenere ottimi punteggi su ROUGE-1, perché include molte parole chiave presenti anche nel gold.

### 5.3 Limiti tipici
- se l’articolo introduce l’argomento con contesto lento o narrativo, Lead-3 può perdere punti chiave presenti dopo;
- può risultare meno “riassuntivo” (tende a copiare, non comprimere);
- può includere dettagli non essenziali solo perché compaiono nelle prime frasi;
- non adatta il contenuto alla lunghezza ideale di un riassunto: prende tre frasi “a prescindere”.

---

## 6. Baseline Extractive 2: TF-IDF + Similarità Coseno
### 6.1 Intuizione
L’obiettivo è selezionare frasi che siano “rappresentative” del contenuto informativo dell’articolo.

La pipeline segue questa logica:
1) si segmenta l’articolo in frasi;
2) ogni frase viene rappresentata con un vettore **TF-IDF** (peso dei termini);
3) si costruisce una rappresentazione globale del documento (o una media/aggregazione);
4) si calcola la **similarità coseno** tra ogni frase e il documento;
5) si selezionano le frasi con similarità più alta (riassunto extractive).

### 6.2 Perché TF-IDF ha senso come baseline
- È semplice, trasparente e riproducibile.
- Tende a privilegiare termini specifici (alta IDF) e quindi concetti distintivi dell’articolo.
- Funziona bene quando il testo ha un “topic” chiaro e il vocabolario distintivo è concentrato in poche frasi.

### 6.3 Limiti tipici e comportamento atteso
- La “rappresentatività lessicale” non coincide sempre con “importanza informativa”: una frase può essere lessicalmente centrale ma poco utile come riassunto.
- Può selezionare frasi simili tra loro → rischio ridondanza.
- Non garantisce coesione narrativa: frasi estratte da punti distanti possono risultare scollegate.
- Se il documento è lungo e copre più sotto-temi, TF-IDF può favorire frasi “medie” piuttosto che quelle realmente decisive.

In sintesi: TF-IDF è una baseline interpretabile, ma spesso meno efficace di Lead-3 in news e meno “fluida” di un modello generativo fine-tunato. :contentReference[oaicite:10]{index=10}

---

## 7. Valutazione: ROUGE + Analisi Qualitativa
La valutazione quantitativa usa ROUGE:
- ROUGE-1 (unigrammi),
- ROUGE-2 (bigrammi),
- ROUGE-L / ROUGE-Lsum (similarità basata su LCS e versione adatta a summarization). :contentReference[oaicite:11]{index=11}

A ROUGE viene affiancata un’analisi qualitativa perché:
- ROUGE misura overlap lessicale, ma non “capisce” bene coerenza, accuratezza o eventuali omissioni importanti;
- i metodi extractive possono ottenere buoni ROUGE copiando porzioni grandi, mentre l’abstractive può riformulare e perdere overlap pur essendo valido.

---

## 8. Risultati (Evaluation subset N=500)
I risultati riportati in relazione mostrano un quadro coerente con la natura dei metodi. :contentReference[oaicite:12]{index=12}

**ROUGE (come da tabella in relazione):** :contentReference[oaicite:13]{index=13}
- **BART**: ROUGE-1 0.4164 | ROUGE-2 0.1968 | ROUGE-L 0.2883 | ROUGE-Lsum 0.3850
- **Lead-3**: ROUGE-1 0.4181 | ROUGE-2 0.1935 | ROUGE-L 0.2649 | ROUGE-Lsum 0.3471
- **TF-IDF**: ROUGE-1 0.3671 | ROUGE-2 0.1538 | ROUGE-L 0.2419 | ROUGE-Lsum 0.3138

**Lettura dei risultati (in breve):**
- Lead-3 è molto competitivo su ROUGE-1 (tipico nel dominio news).
- BART risulta particolarmente forte su ROUGE-2 e metriche legate a struttura/coerenza, coerentemente con la capacità di riformulazione e sintesi.
- TF-IDF resta più debole: utile come baseline “classica”, ma meno adatta a produrre riassunti coesi e informativamente ottimali. :contentReference[oaicite:14]{index=14}

---

## 9. Ambiente di esecuzione
Gli esperimenti sono stati eseguiti in Google Colab con GPU **NVIDIA A100 (40GB)**, coerentemente con i requisiti computazionali del fine-tuning e della generazione con Transformer. :contentReference[oaicite:15]{index=15}

---

## 10. Repository Structure
Struttura consigliata per GitHub:



