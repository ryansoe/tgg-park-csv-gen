# Robsham Theater to O'neill Library

## Directions created manually

Walk out of Robsham Theater and turn left onto Campanella Way.

Keep walking straight until you reach the Commonwealth Avenue Garage.

Follow the sidewalk that curves left around the garage.

Keep going past the Communication Department on your right.

Continue straight and you’ll see O'neill Library on your left. This is your destination!

## Directions created by script

Walking tour from: 42.33793,-71.16845
Destination: 42.33601,-71.16955
Total distance: 280 m
Estimated time: 4 min

Directions:

1. Head W on path for 54 m
2. Turn right onto path for 36 m
3. Turn left onto path for 80 m
4. Continue on path for 110 m

Enjoy your walk!

# 9/11 Memorial Labryinth to Conte Forum

## Directions created manually

Starting on the path from the 9/11 Memorial Labryinth, go straight and turn right onto Linden Lane. On your left, you'll see Bapst Library.

Head straight until you reach Gasson Hall.

Turn left, walk straight, then turn right.

Walk forward until you reach the Boston College Office of Undergraduate Admission.

Turn left, then walk straight until you reach Conte Forum. This is your destination

## Directions created by script

Walking tour from: 42.33704,-71.17169
Destination: 42.33578,-71.16793
Total distance: 0.29 mi
Estimated time: 6 min

Directions:

1. Head S on path for 93 m
   - You'll pass on your right Waban Hill (natural).
   - You'll pass on your right Waban Hill Reservoir (natural).
2. Turn right onto path for 120 m
   - You'll pass on your left WZBC-FM (Newton) (man_made).
3. Turn left onto path for 18 m
4. Turn slight left onto path for 53 m
5. Turn right onto path for 5 m
6. Turn left onto path for 59 m
7. Turn right onto path for 110 m
   - Ahead on your right: Robsham Theater Arts Center (amenity).

Enjoy your walk!

# Report

### What to improve

- **Use names over coordinates**: Manual directions reference buildings and street names (e.g., `Robsham Theater`, `Linden Lane`, `O'neill Library`) rather than coordinates or generic "path".
- **Natural phrasing**: Prefer "Walk/Continue" over cardinal phrases like "Head W"; avoid "onto path" when unnamed.
- **Fewer micro-steps**: Merge tiny zig-zag turns and short segments so the route reads like 3–5 clear steps.
- **Cleaner callouts**: Only mention useful landmarks (campus buildings, parks, libraries, stadiums); avoid technical categories like `man_made`/`natural` and parenthetical suffixes like "(Newton)".
- **Clear arrival**: End with "Arrive at <destination>. This is your destination!".

### Actionable code changes

- **Labels for start/end**

  - In `geocode_point` and `format_tour_text`, prefer names: if inputs are lat/lon, add reverse geocoding to derive a human label (nearest named building/POI or street); allow manual override via new CLI flags `--start-label`, `--end-label`.

- **Step phrasing and templates** (in `build_directions`)

  - Replace cardinal starts: instead of "Head NE on X", use "Walk on X" for the first step and "Continue on X" for same-name continuations.
  - Rename slight turns: in `_turn_phrase`, map slight (<35°) to "bear left/right" and treat very small (<20–25°) as "continue" (no new step).
  - Avoid "onto path": when `_edge_name` falls back to "path", look at the `highway` tag to say "trail", "footpath", or omit the facility type entirely (e.g., "Turn left" vs "Turn left onto path").
  - Add a final arrival line: after steps, append `Arrive at {end.label}. This is your destination!`.

- **Step merging and smoothing**

  - Merge short segments: when a prospective split would create a step < 25–30 m, fold it into the previous step unless the turn is strong (≥ 60–90°).
  - Debounce zig-zags: collapse consecutive alternations of slight left/right within ~15–20 m total length into a single "Continue" step.
  - Geometry simplification: optionally simplify the route `LineString` with a small tolerance (e.g., 2–4 m) before step detection to reduce micro-turns.

- **Callout selection and formatting** (in `select_and_order_pois` and `weave_pois_into_steps`)

  - Filter categories: prefer campus/venue categories (building, library, museum, stadium, academic, chapel); drop generic `natural`, `man_made`, utilities, broadcast facilities.
  - Clean names: strip parenthetical disambiguation (e.g., "(Newton)") and avoid emitting raw OSM feature types in parentheses.
  - Limit density: at most 1 callout per step by default; put the most relevant one (highest score, closest) first.
  - Friendlier copy: format callouts as "You'll pass on your {left/right} <Name>." and omit technical descriptors unless `--callout-style=descriptive` is set.

- **Distance display**

  - Keep total distance/time at the top; add `--hide-step-distances` to suppress per-step distances so steps read like the manual examples.
  - If showing distances, keep meters for short routes; otherwise your current thresholds are fine.

- **Start/egress heuristics**
  - If the start point is within a named building polygon, add an initial unnumbered pre-step like "Walk out of <Building>" and then begin step 1 at the first network segment.
  - If the distance from the raw start point to the first edge exceeds ~10 m, insert a short connector phrase "Walk to <Street/Trail>" rather than a numbered turn.

### CLI additions (backward-compatible)

- `--start-label`, `--end-label`: override names shown in header and arrival line.
- `--hide-step-distances`: remove per-step "for X" text.
- `--max-callouts-per-step` (default 1) and `--callout-style` (`minimal`|`descriptive`).
- `--preferred-categories` and `--blocked-categories`: CSV lists to tune callout filtering.

### Using an LLM API to improve generating directions

- **Where to integrate**

  - Post-process after `build_directions` and POI selection but before `format_tour_text`.
  - Input to the LLM: a compact JSON with steps (name, distance, bearing/turn), POIs (name, side, along_m, category), start/end labels, and style flags.

- **What the LLM should do (bounded scope)**

  - Merge micro-steps and rewrite into natural, concise directions using "Walk/Continue/Turn".
  - Add a clear arrival line.
  - Select ≤1 callout per step with friendly phrasing; drop technical categories; never invent new places.
  - Preserve distances if enabled, otherwise omit per-step distances.

- **Safety constraints (avoid hallucinations)**

  - Provide the full list of allowed names and forbid introducing any not present in input.
  - Forbid fabricating street/building names; if uncertain, fall back to generic phrasing ("Continue").
  - Require structured output (JSON) that maps 1:1 to provided step ids and callout ids.

- **Prompt sketch**

  - System: "You rewrite walking directions for clarity. Only use provided names. Do not fabricate new landmarks or streets. Keep 8-10 steps."
  - User content (JSON): `{ steps: [...], pois: [...], options: { maxSteps: 6, includeDistances: false, tone: "concise" } }`
  - Response format: `{ steps: [{id, text}], callouts: [{stepId, text}], arrival: "..." }`

- **Minimal Python integration example**
  - After computing `steps`, `callouts` and `total_m`:

```python
payload = {
    "steps": [
        {"id": i+1, "name": s.text, "distance_m": s.distance_m}
        for i, s in enumerate(steps)
    ],
    "pois": [
        {"id": i+1, "name": c.name, "side": c.side, "along_m": c.along_m}
        for i, c in enumerate(callouts)
    ],
    "options": {"maxSteps": 10, "includeDistances": False, "tone": "concise"},
}

# pseudo-call
resp = llm_client.chat(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    system="You rewrite walking directions... Only use provided names.",
    user_json=payload,
    response_format={"type": "json_object"},
)

rewritten = json.loads(resp)
lines = [f"{i+1}. {item['text']}" for i, item in enumerate(rewritten["steps"])]
for c in rewritten.get("callouts", []):
    lines.append(f"   - {c['text']}")
if rewritten.get("arrival"):
    lines.append(rewritten["arrival"])
```

- **CLI switches for LLM**

  - `--llm-enabled`: enable LLM post-processing.
  - `--llm-model`, `--llm-temperature`, `--llm-style` (e.g., `concise`, `narrative`).
  - `--llm-max-steps` and `--llm-include-distances` to mirror non-LLM behavior.

- **Caching and determinism**

  - Cache LLM outputs by hashing the input JSON (e.g., under `cache/llm/`) to avoid re-billing on identical routes.
  - Set temperature low (≤0.2) for stable outputs.

# notes

confidence score to know which should be looked at
generates 3 responses using LLM based on complexity (simple, medium, complex)
