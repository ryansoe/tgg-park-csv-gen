## Park Game POI Generator

Generate a CSV of interesting places inside a park using OpenStreetMap via OSMnx/Overpass, suitable for building walking game routes.

### Features

- Finds a park polygon (either a specific park by name or the largest park in a city)
- Pulls POIs (historic, tourism, leisure, amenity, natural, man_made)
- Scores and filters them for interest
- Selects a walkable subset (default ≤ 3 total miles, up to 15 spots)
- Exports a CSV with coordinates and metadata

## Installation

```bash
# from repo root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Usage

Run the generator with either a specific park or a city (to auto-pick the largest park).

### Largest park in a city

```bash
source .venv/bin/activate
python park_game_generator.py --city "Pittsburgh, Pennsylvania" --output pittsburgh_park_pois.csv
```

### Specific park by name

```bash
source .venv/bin/activate
python park_game_generator.py --park "Central Park, New York" --output central_park_pois.csv
python park_game_generator.py --park "Golden Gate Park, San Francisco" --output golden_gate_park_pois.csv
```

### Options

- `--park` string: Specific park name or query string.
- `--city` string: City query; script will pick the largest park inside the city.
- `--output` string: Output CSV path (default `park_pois.csv`).
- `--max-points` int: Max number of POIs to select (default 15).
- `--max-miles` float: Max total walking miles for the greedy route heuristic (default 3.0).
- `--min-park-area-m2` float: Minimum park area when using `--city` (default 300,000 m²).

Note: `--park` and `--city` are mutually exclusive. If neither is provided, the default is the largest park in Pittsburgh, PA.

### Output CSV columns

- `park`: Park display name.
- `name`: POI name (falls back to "Unnamed" if none in OSM).
- `lat`, `lon`: Centroid coordinates of the POI geometry (WGS84).
- `category`: Inferred category (historic, tourism, leisure, amenity, natural, man_made).
- `score`: Interest score assigned by heuristics.
- `has_plaque`: Boolean flag based on tags/inscriptions.
- `wikidata`, `wikipedia`, `image`: Enrichment fields when present.
- `operator`, `website`, `start_date`, `opening_hours`: Commonly useful tags flattened.
- `extra_info`: JSON string of all tags from OSM.
- `osmid`: OSM element identifier (e.g., `('node', 12345)`).

## How the algorithm works (brief)

1. Park polygon
   - If `--park` is provided, geocode the park name and use its polygon.
   - Else, geocode the city, fetch park-like features inside it, compute area, and choose the largest polygon.
2. POI retrieval
   - Query OSM features inside the polygon for keys/values across categories (`historic`, `tourism`, `leisure`, `amenity`, `natural`, `man_made`).
   - Normalize each feature to a row with centroid coords and tags.
3. Scoring
   - Base weights by category (e.g., historic, tourism).
   - Boosts for tokens found in tags/name (e.g., museum, viewpoint, memorial, plaque, heritage, statue, garden, bridge).
   - Penalties for utilitarian tokens (e.g., toilet, parking, bench, waste).
   - Keep rows with positive total score.
4. Selection (≤ 3 miles by default)
   - Sort by score and run a greedy nearest-neighbor: start at the highest score, add the next best nearby (score − distance) while total distance ≤ `--max-miles`, up to `--max-points`. Unnamed POIs are excluded from selection.
5. Export
   - Write selected rows to CSV with the columns listed above.
