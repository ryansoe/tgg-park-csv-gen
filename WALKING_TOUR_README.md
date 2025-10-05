## Walking Tour Generator

Generate turn-by-turn walking directions between two locations and interleave nearby points of interest (POIs) as a concise “guided tour” text file.

### Features

- Geocode start and end (address strings or `lat,lon`)
- Build a walking network and compute the shortest walking route
- Produce human-readable directions (street names, turns, distances)
- Buffer the route into a corridor and fetch POIs from OpenStreetMap
- Score POIs, exclude unnamed features, order them along the path
- Optionally enrich POIs with Google Places ratings/IDs
- Output a tidy `.txt` walkthrough with interleaved POI callouts

## Installation

```bash
# from repo root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Usage

```bash
source .venv/bin/activate
python walking_tour_generator.py --start "START_ADDRESS_OR_LATLON" --end "END_ADDRESS_OR_LATLON" --output path/to/tour.txt
```

### Examples

Addresses:

```bash
python walking_tour_generator.py \
  --start "Union Square, San Francisco" \
  --end "Ferry Building, San Francisco" \
  --output walking_tours/sf_union_to_ferry.txt \
  --max-pois 12 --buffer-meters 100
```

Coordinates:

```bash
python walking_tour_generator.py \
  --start "37.7689922,-122.509449" \
  --end "37.7699405,-122.5104577" \
  --output walking_tours/ggp_cityfields_to_roald_amundsen.txt \
  --max-pois 12 --buffer-meters 100
```

### Options

- `--start` string (required): Start address or `lat,lon`.
- `--end` string (required): End address or `lat,lon`.
- `--output` string: Output `.txt` path (default `walking_tour.txt`).
- `--buffer-meters` int: Half-width corridor to search for POIs (default 100).
- `--max-pois` int: Max POIs to include (default 15).
- `--min-poi-score` float: Minimum POI score to consider (default 1.0).
- `--graph-distance-m` int: Radius (meters) around midpoint to load walking network; auto if omitted.
- `--enable-google-places`: Enrich with Places ratings/IDs.
- `--google-api-key` string: Google Maps/Places API key (or set `GOOGLE_MAPS_API_KEY`).
- `--google-radius-meters` int: Search radius for Google matching (default 150).

### Output

Plain text with:

- Header: start, destination, total distance, estimated time
- Numbered steps with distances and turn instructions
- Interleaved POI callouts like: `- You'll pass on your left Foo Museum (tourism, rated 4.6).`

## How it works (brief)

1. Geocode start/end to WGS84 points (or parse `lat,lon`).
2. Load a walking graph around the midpoint with a radius sized to cover the corridor.
3. Compute the shortest path by `length` and derive readable steps by merging edges with similar names/bearings.
4. Convert the route to a `LineString` and buffer it (in a metric CRS) to make a corridor polygon.
5. Query OSM features inside the corridor, score them for interest, and exclude unnamed POIs.
6. Rank by score and proximity to the route, then order along the path.
7. Weave callouts into the step list and write a `.txt` tour.

## Tips

- Increase `--buffer-meters` (e.g., 120–150) for parks or wide routes; decrease (75–100) in dense city centers.
- Keep `--max-pois` modest (10–15) to avoid cluttered narration on short walks.
- If you enable Google enrichment, export an environment variable: `export GOOGLE_MAPS_API_KEY=...`.
- If you hit an OSMnx warning about bbox coordinate order, it’s informational and safe to ignore.
- If you see the scikit-learn error, install it as shown above.
