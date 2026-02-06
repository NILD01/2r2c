# Analyse & nieuwe start (2R2C)

## Observaties uit het oude project (`old/`)
- **Opsplitsing tslow/tfast**: de huidige workflow schat eerst aparte tijdconstanten uit specifieke segmenten en construeert daaruit 2R2C-parameters. Dit maakt het model gevoelig voor segmentselectie, en forceert de eigenwaarden *vooraf* in plaats van een gezamenlijke fit op de volledige data. `old/r2r2c.py` koppelt de fit bovendien stevig aan de segment-pijplijn en aan het idee dat alle segmenten “geen actieve bronnen” zijn.
- **Latente massatemperatuur**: het huidige `r2r2c.py` lost het initiële massa-niveau op via een 1-parameter least-squares truc per segment. Dat helpt lokaal, maar het is geen expliciet modelleren van interne/warmtestoringen doorheen de tijd.
- **Probleem bij praktische data**: in de ruwe `train.csv` zitten duidelijke warmtebronnen (CV, warmtepomp, apparaten, zon). Deze bronnen worden in de oude aanpak alleen indirect geabsorbeerd via segmentselectie, waardoor het model slecht generaliseert buiten die segmenten.

## Waarom een gezamenlijke fit + warmte‑storing beter werkt
- Met één gezamenlijke fit op alle data hoef je **tfast/tslow niet vooraf te fixeren**. De 2R2C-parameters worden dan *direct* geschat uit de volledige tijdreeks.
- Door een **latente warmte‑storing** toe te voegen (een extra state of input) kan het model op elk moment extra warmte opnemen of afgeven zonder dat de weerstands-/capaciteitsparameters moeten “meeschuiven”.
- Dit zorgt voor een robuustere schatting van de **fysische parameters** (Ria, Rao, Cm), terwijl de verstoring de niet‑gemodelleerde inputs opvangt (zon, CV, bewoners, toestellen, enz.).

## Wat ik in deze nieuwe start heb toegevoegd
- Een nieuw script `fit_2r2c_disturbance.py` dat:
  - `train.zip` rechtstreeks leest;
  - de 2R2C‑parameters **gemeenschappelijk** fit via Kalman‑filter log‑likelihood;
  - een **latente warmte‑storing** (`Qdist`) als extra state meeneemt;
  - optioneel ook het **volume** mee fit, zodat Ci niet meer vastgezet hoeft te worden;
  - een CSV export maakt met gefilterde binnen‑temperatuur en geschatte Qdist.
- Een `requirements.txt` zodat je in één keer de benodigde packages kunt installeren.

## Tips voor je eerste Codex‑workflow
1. **Werk iteratief**: begin met één helder doel (bv. “fit 2R2C + Qdist”), en breid daarna pas uit met solar/HP/CV‑inputs.
2. **Maak beslissingen expliciet**: noteer in een markdown (zoals deze) waarom je keuzes maakt (segmentselectie, noise‑niveau, etc.). Dit helpt enorm bij latere debug.
3. **Gebruik logs en export**: kijk niet enkel naar RMSE, maar ook naar de tijdreeks van Qdist en residualen om te zien of het model fysisch plausibel blijft.
4. **Laat Codex kleine stappen doen**: vraag eerst om een script dat enkel fit + export doet, en pas daarna om visualisaties of extra features.

## Volgende logische stap
- Als deze baseline werkt, kunnen we Qdist vervangen of verklaren door **expliciete inputs** (zon, CV, warmtepomp, interne loads), zodat Qdist kleiner en stabieler wordt.
- Daarna kan je het model inzetten als **predictor** met expliciete control‑inputs.
