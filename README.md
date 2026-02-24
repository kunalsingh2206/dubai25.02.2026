
# Dubai Real Estate Transactions Dashboard (Streamlit)

## What this project does
- Reads the Dubai transactions Excel export (sheet: `Export`)
- Cleans and standardizes it (Ready vs Off-plan, Villa/Apartment only, bedroom whitelist)
- Groups developer names using fuzzy logic + optional manual aliases
- Flags statistical outliers (Size, Amount, PSF) so the dashboard can exclude them by default
- Provides an executive dashboard with:
  - Top filters (date, transaction type bucket, property type)
  - Secondary filters (bedrooms, ranges, community/developer/property)
  - Scenario comparison (up to 4)
  - Hotspots (top 5 by count and by amount)
  - Dry-period analytics (streaks of low-activity days)

## Quick start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Daily update workflow (no uploads)
1. Keep the raw file at:
   `data/raw/transactions.xlsx`
2. Every day, paste new transactions into the same Excel file (append rows).
3. Open the Streamlit app and click **Rebuild cleaned dataset now** (sidebar),
   OR run the ETL script:
```bash
python etl_run.py --raw data/raw/transactions.xlsx --out data/processed/transactions_clean.pkl
```

### Best practice (recommended)
- Never edit the cleaned file manually.
- Keep a single source of truth: the raw Excel export.
- If you host Streamlit on a server, keep the Excel file in a shared folder (OneDrive/SharePoint/Network drive)
  that syncs to the server path.

## Developer grouping
- Canonical names live in `src/config.py` (CANONICAL_DEVELOPERS)
- Manual aliases live in `config/developer_aliases.csv`
  This is the fastest way to fix edge cases when an executive spots a mismatch.

## Notes / assumptions
- `Times Sold` is filled with 0 where missing (many off-plan rows).
- "Dry period" is computed from a complete daily time series and is defined as
  consecutive days where sales are <= x% of the peak day (max or 80th percentile).
