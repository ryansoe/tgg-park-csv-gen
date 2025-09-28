#!/usr/bin/env python3

import argparse
import os
import re
import sys
from typing import List, Optional

from park_game_generator import process_park


def _read_parks_from_file(path: str) -> List[str]:
    parks: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parks.append(s)
    return parks


def _slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", name)
    s = s.strip("_")
    return s.lower() or "park"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch run park POI generator for multiple parks")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--parks", nargs="+", help="One or more park query strings")
    src.add_argument("--parks-file", type=str, help="Path to a file with one park per line")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to write CSVs into")
    parser.add_argument("--max-points", type=int, default=15, help="Max number of POIs to select per park")
    parser.add_argument("--max-miles", type=float, default=3.0, help="Max total walking miles per park")
    parser.add_argument("--min-park-area-m2", type=float, default=300_000.0, help="Minimum area when auto-picking parks (not used here)")
    parser.add_argument("--distance-penalty", type=float, default=5.0, help="Multiplier in rank = score − (penalty × miles)")
    parser.add_argument("--enable-google-places", action="store_true", help="Enrich POIs using Google Places (adds ratings and IDs)")
    parser.add_argument("--google-api-key", type=str, default=None, help="Google Maps/Places API key (or set GOOGLE_MAPS_API_KEY)")
    parser.add_argument("--google-radius-meters", type=int, default=150, help="Search radius for Places matching (meters)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    parks: List[str] = []
    if args.parks:
        parks = list(args.parks)
    elif args.parks_file:
        parks = _read_parks_from_file(args.parks_file)

    if not parks:
        print("No parks provided", file=sys.stderr)
        return 2

    successes = 0
    for park_query in parks:
        slug = _slugify(park_query)
        out_csv = os.path.join(args.output_dir, f"{slug}.csv")
        try:
            print(f"[Batch] Processing: {park_query} → {out_csv}")
            process_park(
                park=park_query,
                city=None,
                output_csv=out_csv,
                max_points=args.max_points,
                max_total_miles=args.max_miles,
                min_park_area_m2=args.min_park_area_m2,
                enable_google_places=args.enable_google_places,
                google_api_key=args.google_api_key,
                google_radius_meters=args.google_radius_meters,
                distance_penalty=args.distance_penalty,
            )
            successes += 1
        except Exception as e:
            print(f"[Batch] Error processing '{park_query}': {e}", file=sys.stderr)

    print(f"[Batch] Completed. {successes}/{len(parks)} succeeded.")
    return 0 if successes == len(parks) else 1


if __name__ == "__main__":
    raise SystemExit(main())


