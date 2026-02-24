
"""
Command-line runner for the ETL.

Use this if you want a fast daily refresh without opening the Streamlit app.

Example:
  python etl_run.py --raw data/raw/transactions.xlsx --out data/processed/transactions_clean.pkl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.etl import etl_run


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="Path to raw XLSX file")
    p.add_argument("--out", required=True, help="Output path (parquet/pkl/csv)")
    p.add_argument("--aliases", default="config/developer_aliases.csv", help="Developer alias mapping CSV (optional)")
    args = p.parse_args()

    aliases = args.aliases if Path(args.aliases).exists() else None
    audit = etl_run(raw_xlsx_path=args.raw, out_path=args.out, developer_alias_csv=aliases)
    print("ETL complete.")
    for k, v in audit.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
