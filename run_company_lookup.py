from __future__ import annotations

import argparse
import json
from src.pipelines.resolve_company_industry import resolve_company_industry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve company CAEN/industry from ONRC datasets with Termene fallback.")
    parser.add_argument("company_name", help="Company name to search")
    parser.add_argument("--od-firme", dest="od_firme", default=None, help="Path to od_firme.csv")
    parser.add_argument("--od-caen", dest="od_caen", default=None, help="Path to od_caen_autorizat.csv")
    parser.add_argument("--n-caen", dest="n_caen", default=None, help="Path to n_caen.csv")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = resolve_company_industry(
        company_name=args.company_name,
        od_firme_csv_path=args.od_firme,
        od_caen_autorizat_csv_path=args.od_caen,
        n_caen_csv_path=args.n_caen,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

