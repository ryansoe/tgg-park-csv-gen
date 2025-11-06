#!/usr/bin/env python3

"""
Verbose version of batch_run_campus_tours.py with progress tracking.
Shows detailed progress for debugging slow operations.
"""

import argparse
import csv
import os
import re
import sys
import time
from itertools import combinations
from typing import List, Optional, Tuple

from walking_tour_generator import run as run_walking_tour


def _read_campus_pois(csv_path: str, school_filter: Optional[str] = None) -> List[Tuple[str, str, str, str, str]]:
    """Read campus POIs from CSV and optionally filter by school name.
    
    Returns list of (school_name, coordinates, location_name, lat, lon) tuples.
    """
    pois = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            school = row.get("School Name", "").strip()
            coords = row.get("Coordinates", "").strip()
            location = row.get("Location Name", "").strip()
            
            if not school or not coords or not location:
                continue
            
            # Skip if filtering and doesn't match
            if school_filter and school.lower() != school_filter.lower():
                continue
            
            # Parse coordinates
            try:
                parts = coords.split(",")
                if len(parts) != 2:
                    continue
                lat = parts[0].strip()
                lon = parts[1].strip()
                pois.append((school, coords, location, lat, lon))
            except Exception:
                continue
    
    return pois


def _slugify(name: str) -> str:
    """Convert a name into a filesystem-safe slug."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", name)
    s = s.strip("_")
    return s.lower() or "location"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch generate walking tours between campus POIs (VERBOSE)"
    )
    parser.add_argument(
        "--campus-csv",
        type=str,
        default="campus_pois.csv",
        help="Path to campus POIs CSV file",
    )
    parser.add_argument(
        "--school",
        type=str,
        required=True,
        help="School name to filter POIs (e.g., 'Boston College')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="walking_tours",
        help="Directory to write tour files into",
    )
    parser.add_argument(
        "--mode",
        choices=["consecutive", "all-pairs"],
        default="consecutive",
        help="Tour generation mode: consecutive (each to next) or all-pairs",
    )
    parser.add_argument(
        "--max-pois",
        type=int,
        default=10,
        help="Max number of POIs to include per tour",
    )
    parser.add_argument(
        "--buffer-meters",
        type=int,
        default=100,
        help="Half-width corridor for POI search",
    )
    parser.add_argument(
        "--llm-enabled",
        action="store_true",
        default=True,
        help="Enable LLM post-processing (default: True)",
    )
    parser.add_argument(
        "--llm-complexity",
        choices=["simple", "medium", "complex"],
        default="medium",
        help="LLM complexity level",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use",
    )
    parser.add_argument(
        "--enable-google-places",
        action="store_true",
        help="Enrich POIs with Google Places",
    )
    parser.add_argument(
        "--google-api-key",
        type=str,
        default=None,
        help="Google API key (or set GOOGLE_MAPS_API_KEY env var)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Timeout per tour in seconds (default: 180)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    
    # Read and filter POIs
    print(f"📁 Reading POIs from {args.campus_csv}...")
    pois = _read_campus_pois(args.campus_csv, school_filter=args.school)
    
    if not pois:
        print(f"❌ Error: No POIs found for school '{args.school}'", file=sys.stderr)
        return 1
    
    print(f"✓ Found {len(pois)} POIs for {args.school}")
    for i, (school, coords, location, lat, lon) in enumerate(pois, 1):
        print(f"  {i}. {location} ({coords})")
    
    # Create output directory
    school_slug = _slugify(args.school)
    output_dir = os.path.join(args.output_dir, school_slug)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📂 Output directory: {output_dir}")
    
    # Generate tours based on mode
    tours = []
    if args.mode == "consecutive":
        # Generate tours between consecutive POIs
        for i in range(len(pois) - 1):
            start_poi = pois[i]
            end_poi = pois[i + 1]
            tours.append((start_poi, end_poi))
    else:  # all-pairs
        # Generate tours between all pairs of POIs
        for start_poi, end_poi in combinations(pois, 2):
            tours.append((start_poi, end_poi))
    
    print(f"\n🚶 Generating {len(tours)} tours in '{args.mode}' mode...")
    
    successes = 0
    failures = []
    
    for idx, (start_poi, end_poi) in enumerate(tours, 1):
        start_school, start_coords, start_location, start_lat, start_lon = start_poi
        end_school, end_coords, end_location, end_lat, end_lon = end_poi
        
        # Create output filename with numeric prefix
        start_slug = _slugify(start_location)
        end_slug = _slugify(end_location)
        output_file = os.path.join(output_dir, f"{idx}_{start_slug}_to_{end_slug}.txt")
        
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(tours)}] {start_location} → {end_location}")
        print(f"Output: {output_file}")
        print(f"{'='*60}")
        
        # Build command line arguments for walking_tour_generator
        tour_args = [
            "--start", start_coords,
            "--end", end_coords,
            "--start-label", start_location,
            "--end-label", end_location,
            "--output", output_file,
            "--max-pois", str(args.max_pois),
            "--buffer-meters", str(args.buffer_meters),
        ]
        
        if args.llm_enabled:
            tour_args.extend([
                "--llm-enabled",
                "--llm-complexity", args.llm_complexity,
                "--llm-model", args.llm_model,
            ])
        
        if args.enable_google_places:
            tour_args.append("--enable-google-places")
            if args.google_api_key:
                tour_args.extend(["--google-api-key", args.google_api_key])
        
        start_time = time.time()
        try:
            print(f"⏳ Starting tour generation...")
            result = run_walking_tour(tour_args)
            elapsed = time.time() - start_time
            
            if result == 0:
                successes += 1
                print(f"✓ Success (took {elapsed:.1f}s)")
            else:
                failures.append((idx, start_location, end_location, f"Exit code {result}"))
                print(f"✗ Failed with code {result} (took {elapsed:.1f}s)", file=sys.stderr)
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user after {time.time() - start_time:.1f}s")
            print(f"\nProgress: {successes}/{idx} tours completed successfully")
            return 130
        except Exception as e:
            elapsed = time.time() - start_time
            failures.append((idx, start_location, end_location, str(e)))
            print(f"✗ Error after {elapsed:.1f}s: {e}", file=sys.stderr)
    
    print(f"\n{'='*60}")
    print(f"📊 Batch Summary")
    print(f"{'='*60}")
    print(f"✓ Successful: {successes}/{len(tours)}")
    print(f"✗ Failed: {len(failures)}/{len(tours)}")
    print(f"📂 Output: {output_dir}")
    
    if failures:
        print(f"\n❌ Failed tours:")
        for idx, start, end, error in failures:
            print(f"  [{idx}] {start} → {end}")
            print(f"      Error: {error}")
    
    return 0 if successes == len(tours) else 1


if __name__ == "__main__":
    raise SystemExit(main())

