# Bank Statement Processor (Procesare Extras Cont)

This application helps you automatically process bank statements (PDFs) into clear, organized reports. It takes a raw bank PDF and gives you clean data + budget insights.

## What It Does for You

Instead of manually copying transactions into Excel, this tool automatically:

1.  **Reads your PDF bank statement**.
2.  **Cleans up descriptions** (e.g., turns "COST COFFEE S.R.L - POS 123" into "COST COFFEE").
3.  **Identifies the merchant** and looks up their official business activity (CAEN code) to know if they sell food, gas, or IT services.
4.  **Categorizes expenses** automatically (Food, Transport, Utilities, etc.).
5.  **Creates a monthly summary** showing exactly how much you spent vs. earned in each category.

## How to Use It (Very Simple)

You don't need to be a programmer. Just follow these 3 steps:

1.  **Run the application** (double-click the start script or run the command below).
2.  A window will pop up asking for your **PDF Statement**: select your file.
3.  Another window will confirm where to save the results: select any **Output Folder** you like.

That's it! The app will process everything and save several Excel-friendly CSV files in your chosen folder.

## Key Files You Will Receive

In your output folder, look for these important files:

-   **`monthly_budget_categories.csv`** → **Best for analysis!** Shows total spending per month for Food, Transport, etc.
-   **`monthly_budget_overview.csv`** → Simple high-level summary (Income vs. Expenses per month).
-   **`transactions_budget_categorized.csv`** → Every single transaction with its assigned category and merchant details.
-   **`pipeline_quality_report.csv`** → Tells you if any PDF lines couldn't be read clearly.

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
-   `pdf_pipeline/` - Core logic modules (extraction, normalization, enrichment).
-   `pdf_pipeline/project_pipeline.py` - Main orchestrator connecting all steps.
-   `tests/` - Automated tests to ensure accuracy.

### External Data

The app uses generic open datasets for merchant lookups in the `DatasetsCAEN/` folder. If these files are missing, the detailed merchant lookup step will simply be skipped or produce warnings.

### Running Tests

```powershell
python -m unittest discover -s tests -p "test_*.py"
```
