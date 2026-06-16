import argparse
import json
from pathlib import Path


def extract_pairs(run_data: dict) -> list[dict]:
    unique_pairs = set()

    for date_payload in run_data.values():
        if not isinstance(date_payload, dict):
            continue

        for section_name in ("labels", "recommendations"):
            rows = date_payload.get(section_name, [])
            if not isinstance(rows, list):
                continue

            for row in rows:
                if not isinstance(row, dict):
                    continue
                ma_id = row.get("ma_id")
                client_id = row.get("client_id")
                if ma_id is None or client_id is None:
                    continue
                unique_pairs.add((str(ma_id), str(client_id)))

    # Stable output order for reproducibility
    return [{"ma": ma_id, "client": client_id} for ma_id, client_id in sorted(unique_pairs)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract unique ma/client pairs from an analysis run file."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to analysis_run_XXXX.json file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output path for pairs JSON",
    )
    args = parser.parse_args()

    with args.input_file.open("r", encoding="utf-8") as f:
        run_data = json.load(f)

    pairs = extract_pairs(run_data)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(pairs)} pairs to {args.output}")
    else:
        print(json.dumps(pairs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
