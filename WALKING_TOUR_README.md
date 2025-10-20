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

# If you see "scikit-learn must be installed to search an unprojected graph":
pip install scikit-learn
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

LLM-Enhanced (Simple):

```bash
export OPENAI_API_KEY="your-api-key-here"
python walking_tour_generator.py \
  --start "Union Square, San Francisco" \
  --end "Ferry Building, San Francisco" \
  --output walking_tours/sf_union_to_ferry_simple.txt \
  --max-pois 10 \
  --llm-enabled \
  --llm-complexity simple
```

LLM-Enhanced (Complex Narrative):

```bash
export OPENAI_API_KEY="your-api-key-here"
python walking_tour_generator.py \
  --start "Union Square, San Francisco" \
  --end "Ferry Building, San Francisco" \
  --output walking_tours/sf_union_to_ferry_tour.txt \
  --max-pois 12 \
  --llm-enabled \
  --llm-complexity complex
```

### Options

#### Basic Options

- `--start` string (required): Start address or `lat,lon`.
- `--end` string (required): End address or `lat,lon`.
- `--output` string: Output `.txt` path (default `walking_tour.txt`).
- `--buffer-meters` int: Half-width corridor to search for POIs (default 100).
- `--max-pois` int: Max POIs to include (default 15).
- `--min-poi-score` float: Minimum POI score to consider (default 1.0).
- `--graph-distance-m` int: Radius (meters) around midpoint to load walking network; auto if omitted.

#### Label Customization

- `--start-label` string: Override display name for start location.
- `--end-label` string: Override display name for end location.
- `--hide-step-distances`: Omit per-step distance text.

#### POI Callout Options

- `--max-callouts-per-step` int: Max POI callouts per step (default 1).
- `--callout-style` {minimal,descriptive}: Callout verbosity (default minimal).
- `--preferred-categories` string: CSV list of preferred POI categories.
- `--blocked-categories` string: CSV list of blocked POI categories.

#### Google Places Integration

- `--enable-google-places`: Enrich with Places ratings/IDs.
- `--google-api-key` string: Google Maps/Places API key (or set `GOOGLE_MAPS_API_KEY`).
- `--google-radius-meters` int: Search radius for Google matching (default 150).

#### LLM Post-Processing

- `--llm-enabled`: Enable LLM post-processing of directions for more natural language.
- `--llm-model` string: LLM model to use (default `gpt-4o-mini`).
- `--llm-temperature` float: LLM temperature 0-1 (default 0.2).
- `--llm-complexity` {simple,medium,complex}: Complexity level (default simple).
  - **simple**: Conversational, concise directions like a friend giving directions
  - **medium**: Moderate detail with helpful context and key landmarks
  - **complex**: Rich narrative tour with vivid descriptions and interesting details
- `--llm-max-steps` int: Max steps for LLM to produce (default 10).
- `--llm-include-distances`: Include per-step distances in LLM output.

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
- If you hit an OSMnx warning about bbox coordinate order, it's informational and safe to ignore.
- If you see the scikit-learn error, install it as shown above.

### LLM Feature Tips

- **Setup**: Install the OpenAI package with `pip install openai` and set your API key: `export OPENAI_API_KEY="sk-..."`.
- **Caching**: LLM responses are automatically cached in `cache/llm/` to avoid redundant API calls and costs.
- **Complexity Levels**:
  - Use **simple** for quick, friendly directions (like a local giving directions)
  - Use **medium** for helpful context with key landmarks (good for tourists)
  - Use **complex** for rich narrative tours (great for leisurely walks with interesting details)
- **Cost**: Simple/medium modes typically use fewer tokens than complex. Use `gpt-4o-mini` (default) for lower costs.
- **Quality**: The LLM only uses landmarks and places found by OSM/Google - it won't hallucinate locations.
- **Fallback**: If the LLM call fails or API key is missing, the generator falls back to rule-based directions automatically.
