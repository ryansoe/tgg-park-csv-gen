## Campus Walking Tours Batch Generator

Generate walking tours between campus POIs using the `batch_run_campus_tours.py` script. This tool reads POIs from `campus_pois.csv` and generates turn-by-turn walking directions with LLM-enhanced narrative.

### Quick Start

#### Boston College Tours (Consecutive Mode)

Generate walking tours between consecutive POIs:

```bash
source .venv/bin/activate
python3 batch_run_campus_tours.py \
  --school "Boston College" \
  --mode consecutive \
  --llm-enabled \
  --llm-complexity medium \
  --max-pois 10
```

This will:
- Read all Boston College POIs from `campus_pois.csv`
- Generate 9 tours (10 POIs = 9 consecutive pairs)
- Output to `walking_tours/boston_college/`
- Use LLM with medium complexity for natural language directions

#### All-Pairs Mode

Generate tours between ALL pairs of POIs (combinatorial):

```bash
python3 batch_run_campus_tours.py \
  --school "Boston College" \
  --mode all-pairs \
  --llm-enabled \
  --llm-complexity medium
```

⚠️ **Warning**: For n POIs, this generates n×(n-1)/2 tours. 10 POIs = 45 tours!

### Options

#### Required

- `--school NAME`: School name to filter POIs (e.g., "Boston College", "Cal Poly", "Columbia University")

#### Optional

- `--campus-csv PATH`: Path to campus POIs CSV (default: `campus_pois.csv`)
- `--output-dir PATH`: Output directory (default: `walking_tours`)
- `--mode {consecutive,all-pairs}`: Tour generation mode (default: `consecutive`)
- `--max-pois N`: Max POIs to include per tour (default: 10)
- `--buffer-meters N`: Half-width corridor for POI search (default: 100)

#### LLM Options

- `--llm-enabled`: Enable LLM post-processing (default: True in script)
- `--llm-complexity {simple,medium,complex}`: Complexity level (default: `medium`)
  - **simple**: Concise, friendly directions
  - **medium**: Moderate detail with helpful context
  - **complex**: Rich narrative tour
- `--llm-model NAME`: LLM model to use (default: `gpt-4o-mini`)

#### Google Places Options

- `--enable-google-places`: Enrich POIs with Google ratings
- `--google-api-key KEY`: Google API key (or set `GOOGLE_MAPS_API_KEY`)

### Examples

#### Cal Poly Tours

```bash
python3 batch_run_campus_tours.py \
  --school "Cal Poly" \
  --mode consecutive \
  --llm-complexity simple
```

Output: `walking_tours/cal_poly/alex_g_spanos_stadium_to_cal_poly_art_gallery.txt`, etc.

#### Columbia University with Google Places

```bash
export GOOGLE_MAPS_API_KEY="your-api-key-here"
python3 batch_run_campus_tours.py \
  --school "Columbia University" \
  --mode consecutive \
  --enable-google-places \
  --llm-complexity complex
```

### Output Structure

```
walking_tours/
  boston_college/
    robsham_theater_arts_center_to_o_neill_library.txt
    o_neill_library_to_saint_ignatius_statue.txt
    saint_ignatius_statue_to_gasson_quad.txt
    ...
  cal_poly/
    alex_g_spanos_stadium_to_cal_poly_art_gallery.txt
    ...
  columbia_university/
    butler_library_to_the_sundial.txt
    ...
```

### Available Schools in campus_pois.csv

1. **Boston College** (10 POIs)
2. **Cal Poly** (9 POIs)
3. **Cal State San Bernardino** (9 POIs)
4. **Chico State** (9 POIs)
5. **Columbia University** (10 POIs)
6. **James Madison University** (9 POIs)

### CSV Format

The `campus_pois.csv` file has this structure:

```csv
School Name,Coordinates,Location Name,
Boston College,"42.337930214053415, -71.16845342549557",Robsham Theater Arts Center,
Boston College,"42.33601326608864, -71.16955352468717",O'Neill Library,
...
```

### Prerequisites

1. **Virtual Environment**: Activate the venv first
   ```bash
   source .venv/bin/activate
   ```

2. **OpenAI API Key** (for LLM features):
   ```bash
   export OPENAI_API_KEY="sk-your-key-here"
   ```

3. **Google API Key** (optional, for Places enrichment):
   ```bash
   export GOOGLE_MAPS_API_KEY="your-key-here"
   ```

### How It Works

1. **Read CSV**: Filters POIs by school name
2. **Generate Pairs**: Creates tour pairs based on mode (consecutive or all-pairs)
3. **For Each Tour**:
   - Geocode start/end coordinates
   - Load walking network from OpenStreetMap
   - Compute shortest walking route
   - Find nearby POIs along the route
   - Generate turn-by-turn directions
   - Use LLM to rewrite into natural language (if enabled)
   - Save to `.txt` file

### Caching

- **LLM Responses**: Cached in `cache/llm/` to avoid redundant API calls
- **OSM Data**: Cached by OSMnx in `cache/` directory
- **Google Places**: Cached in `cache/google_places/`

### Troubleshooting

#### "No POIs found for school"

Make sure the school name matches exactly (case-insensitive). Check `campus_pois.csv` for the exact spelling.

#### LLM not working

- Ensure `OPENAI_API_KEY` is set
- Install openai package: `pip install openai`
- Check `cache/llm/` for cached responses

#### Tours are very short

Some campus POIs are very close together. This is normal for compact campuses. The generator will still create meaningful directions.

#### Rate limiting

If generating many tours, you might hit API rate limits. The script continues on errors and reports success/failure counts at the end.

### Manual Single Tour Generation

To generate a single tour manually:

```bash
python3 walking_tour_generator.py \
  --start "42.337930214053415,-71.16845342549557" \
  --end "42.33601326608864,-71.16955352468717" \
  --start-label "Robsham Theater Arts Center" \
  --end-label "O'Neill Library" \
  --output walking_tours/custom_tour.txt \
  --max-pois 10 \
  --llm-enabled \
  --llm-complexity medium
```

### Cost Estimation

Using `gpt-4o-mini` (default):
- **Simple/Medium**: ~$0.01-0.02 per tour
- **Complex**: ~$0.02-0.04 per tour

For 10 Boston College POIs (9 consecutive tours): ~$0.10-0.20 total

### Performance

- **Per Tour**: 30-60 seconds (depending on LLM caching)
- **9 Boston College Tours**: ~5-10 minutes
- **45 All-Pairs Tours**: ~30-45 minutes

### Next Steps

1. Generate tours for other schools:
   ```bash
   python3 batch_run_campus_tours.py --school "Cal Poly" --mode consecutive
   ```

2. Try different complexity levels to see what works best for your use case

3. Combine with Google Places enrichment for richer POI descriptions

4. Customize the walking_tour_generator.py parameters via the batch script

