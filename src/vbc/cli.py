"""Command-line interface for tabular VBC analyses."""

import argparse
import csv
import json
from pathlib import Path

from .core import transform_endpoints


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vector-Based Comparison of endpoint columns")
    parser.add_argument("csv_file", type=Path, help="CSV file with one endpoint per numeric column")
    parser.add_argument("--primary", required=True, help="name of the primary endpoint column")
    parser.add_argument(
        "--id-column",
        help="optional participant-ID column; checked for missing/duplicate IDs and excluded from VBC",
    )
    parser.add_argument(
        "--preprocess",
        choices=("none", "l2", "minmax", "zscore"),
        default="none",
    )
    parser.add_argument("--output", type=Path, help="optional JSON output file")
    return parser


def main() -> None:
    args = _parser().parse_args()
    with args.csv_file.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("CSV file has no header")
        if any(not name.strip() for name in reader.fieldnames):
            raise SystemExit("CSV column names must not be empty")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise SystemExit("CSV column names must be unique")
        if args.id_column and args.id_column not in reader.fieldnames:
            raise SystemExit(f"ID column {args.id_column!r} was not found")
        if args.primary == args.id_column:
            raise SystemExit("the primary endpoint cannot also be the ID column")
        endpoint_names = [name for name in reader.fieldnames if name != args.id_column]
        columns = {name: [] for name in endpoint_names}
        participant_ids = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise SystemExit(f"incorrect number of CSV fields in row {row_number}")
            if args.id_column:
                participant_id = (row[args.id_column] or "").strip()
                if not participant_id:
                    raise SystemExit(f"missing participant ID in row {row_number}")
                participant_ids.append(participant_id)
            for name in endpoint_names:
                try:
                    columns[name].append(float(row[name]))
                except (TypeError, ValueError) as error:
                    raise SystemExit(f"non-numeric value in {name!r}, row {row_number}") from error
        if len(participant_ids) != len(set(participant_ids)):
            raise SystemExit("participant IDs must be unique")

    try:
        results = transform_endpoints(columns, args.primary, preprocessing=args.preprocess)
    except (ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error
    payload = {
        "primary": args.primary,
        "preprocessing": args.preprocess,
        "observation_count": len(next(iter(columns.values()), [])),
        "endpoints": {
            name: {
                "cosine_similarity": result.cosine_similarity,
                "angle_degrees": result.angle_degrees,
                "parallel": result.parallel.tolist(),
                "perpendicular": result.perpendicular.tolist(),
            }
            for name, result in results.items()
        },
    }
    if args.id_column:
        payload["participant_ids"] = participant_ids
    text = json.dumps(payload, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
