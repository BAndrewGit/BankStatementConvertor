# Bank Statement Processor (Procesare Extras Cont)

This application processes BT bank statements (PDF) into cleaned transactions, category distributions, and a model-ready feature dataset.

## What It Does for You

Instead of manually copying transactions into Excel, this tool automatically:

1.  **Extracts transactions from PDF text** (no OCR).
2.  **Classifies transaction type** (card purchase, fee, transfer, ATM, blocked, unknown).
3.  **Extracts and normalizes merchants** deterministically.
4.  **Maps merchants to expense areas** (Food, Housing, Transport, etc.).
5.  **Builds final model input features** and a run quality report.

## How to Use It (Very Simple)

You don't need to be a programmer. Just follow these 3 steps:

1.  **Run the application** (double-click the start script or run the command below).
2.  A window will pop up asking for your **PDF Statement**: select your file.
3.  Another window will confirm where to save the results: select any **Output Folder** you like.

That's it! The app will process everything and save several Excel-friendly CSV files in your chosen folder.

## Key Files You Will Receive

In your output folder, look for these important files:

-   **`final_dataset.csv`** -> One-row feature vector with the exact model columns.
-   **`transactions_classified.csv`** -> Cleaned and classified transaction-level output.
-   **`run_report.json`** -> Run summary (counts, unknowns, latency) + quality metrics.

---

## Technical Details (For Developers)

### Installation

Requires Python 3.10+.

```powershell
python -m pip install -r requirements.txt
```

### Running the App

```powershell
python main.py
```

### Project Structure

-   `main.py` - The GUI entry point.
-   `src/` - Core processing modules (ingestion, classification, features, pipelines).
-   `tests/` - Automated tests to ensure accuracy.

### Active Flow

Single flow used in production:

1. `main.py` opens file/folder selectors.
2. `src/pipelines/run_end_to_end.py` orchestrates parse -> classify -> map -> feature build.
3. Results are written in the selected output folder.

### Running Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

### ONRC + CAEN Company Lookup

Dataset-first lookup for company enrichment (with optional Termene fallback) is available via:

```powershell
python run_company_lookup.py "AUCHAN"
```

Pipeline used:

1. Search company in `DatasetsCAEN/od_firme.csv`.
2. Extract `CUI` and `COD_INMATRICULARE`.
3. Join authorized CAEN from `DatasetsCAEN/od_caen_autorizat.csv`.
4. Join CAEN nomenclature from `DatasetsCAEN/n_caen.csv`.
5. Map CAEN to industry with explicit overrides (`5610`, `6201`, `4711`).
6. If ONRC match misses, fallback to Termene API when Termene credentials are configured.

Termene setup (optional, fallback only):

`.env` example:

```text
TERMENE_API_URL=https://api.termene.ro/v2
TERMENE_USERNAME=your_username
TERMENE_PASSWORD=your_password
TERMENE_SCHEMA_KEY=your_schema_key
TERMENE_ENABLE_FALLBACK=1
```

Or in Windows environment variables:

```powershell
setx TERMENE_API_URL "https://api.termene.ro/v2"
setx TERMENE_USERNAME "YOUR_TERMENE_USERNAME"
setx TERMENE_PASSWORD "YOUR_TERMENE_PASSWORD"
setx TERMENE_SCHEMA_KEY "YOUR_TERMENE_SCHEMA_KEY"
setx TERMENE_ENABLE_FALLBACK "1"
```

Important: the normal flow is dataset-first (`od_firme.csv`) and API calls are attempted only when local matching fails.

