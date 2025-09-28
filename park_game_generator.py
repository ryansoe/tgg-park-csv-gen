#!/usr/bin/env python3

import argparse
import json
import sys
import os
import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
import requests

# OSMnx / Overpass configuration
ox.settings.use_cache = True
ox.settings.log_console = False
# Use modern settings to avoid Overpass errors
try:
    ox.settings.requests_timeout = 180  # seconds
except Exception:
    ox.settings.timeout = 180  # fallback for older versions
ox.settings.overpass_rate_limit = True
# Set Overpass resource limits explicitly
for attr, val in (
    ("overpass_memory", "512Mi"),
    ("overpass_element_limit", "2000000"),
):
    try:
        setattr(ox.settings, attr, val)
    except Exception:
        pass


@dataclass
class POI:
    osmid: str
    name: str
    lat: float
    lon: float
    category: str
    tags: Dict[str, str] = field(default_factory=dict)
    score: float = 0.0


def _largest_polygon(geom) -> Polygon:
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(list(geom.geoms), key=lambda p: p.area)
    return geom.convex_hull


def _pick_name(row: pd.Series) -> str:
    for key in ["name", "official_name", "ref", "alt_name"]:
        val = row.get(key)
        if val is not None and not pd.isna(val):
            s = str(val).strip()
            if s:
                return s
    return "Unnamed"


def get_park_polygon_by_name(park_query: str) -> Tuple[str, Polygon]:
    gdf = ox.geocode_to_gdf(park_query)
    if gdf.empty:
        raise ValueError(f"Could not geocode park: {park_query}")
    geom = gdf.iloc[0].geometry
    polygon = _largest_polygon(geom)
    disp = gdf.iloc[0].get("display_name")
    name = _pick_name(gdf.iloc[0]) if pd.isna(disp) or disp is None else str(disp)
    return name, polygon


def get_largest_park_in_city(city_query: str, min_area_m2: float = 300_000.0) -> Tuple[str, Polygon]:
    """Find the largest park-like polygon within a city boundary."""
    city_gdf = ox.geocode_to_gdf(city_query)
    if city_gdf.empty:
        raise ValueError(f"Could not geocode city: {city_query}")
    city_poly = _largest_polygon(city_gdf.iloc[0].geometry)

    tags = {
        "leisure": ["park", "garden", "nature_reserve"],
        "boundary": ["national_park"],
        "landuse": ["recreation_ground"],
    }
    gdf_func = getattr(ox, "features_from_polygon", getattr(ox, "geometries_from_polygon"))
    parks_gdf = gdf_func(city_poly, tags)
    if parks_gdf.empty:
        raise ValueError(f"No parks found in {city_query}")

    # Compute areas in meters using a projected CRS
    parks_polys = parks_gdf[~parks_gdf.geometry.is_empty].copy()
    # Keep only polygonal geometries for area sorting
    parks_polys = parks_polys[parks_polys.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if parks_polys.empty:
        raise ValueError(f"No polygonal parks found in {city_query}")

    parks_polys = parks_polys.to_crs(3857)
    parks_polys["area_m2"] = parks_polys.geometry.area

    # Filter out tiny parks
    parks_polys = parks_polys[parks_polys["area_m2"] >= min_area_m2].copy()
    if parks_polys.empty:
        # fall back to largest regardless of area
        parks_polys = parks_gdf.to_crs(3857)
        parks_polys["area_m2"] = parks_polys.geometry.area

    largest = parks_polys.sort_values("area_m2", ascending=False).iloc[0]
    geom = largest.geometry
    polygon = _largest_polygon(geom)
    # Reproject back to WGS84 if needed
    if parks_polys.crs and getattr(parks_polys.crs, "to_epsg", lambda: None)() != 4326:
        polygon = gpd.GeoSeries([polygon], crs=parks_polys.crs).to_crs(4326).iloc[0]

    name = _pick_name(largest)
    if not name or name == "Unnamed":
        name = f"Largest Park in {city_query}"
    return name, polygon


def build_category_tag_filters() -> List[Tuple[str, Dict[str, List[str]]]]:
    return [
        ("historic", {"historic": [
            "monument", "memorial", "archaeological_site", "wayside_cross", "wayside_shrine", "castle", "ruins", "battlefield", "heritage"
        ]}),
        ("tourism", {"tourism": ["museum", "viewpoint", "information", "artwork", "attraction"]}),
        ("leisure", {"leisure": ["garden", "nature_reserve"]}),
        ("amenity", {"amenity": ["fountain", "arts_centre", "community_centre", "ranger_station", "information", "bbq", "public_bookcase"]}),
        ("natural", {"natural": ["tree", "wood", "peak", "cliff", "water", "spring"]}),
        ("man_made", {"man_made": ["tower", "bridge", "obelisk", "lighthouse"]}),
    ]


def query_pois_within_polygon(polygon: Polygon) -> pd.DataFrame:
    tag_dict: Dict[str, List[str]] = {}
    for _, tags in build_category_tag_filters():
        for key, values in tags.items():
            tag_dict.setdefault(key, []).extend(values)
    gdf_func = getattr(ox, "features_from_polygon", getattr(ox, "geometries_from_polygon"))
    gdf = gdf_func(polygon, tags=tag_dict)
    if gdf.empty:
        return pd.DataFrame(columns=["osmid", "name", "lat", "lon", "category", "tags", "wikidata", "wikipedia", "image", "has_plaque"]) 

    records: List[Dict[str, object]] = []
    for idx, row in gdf.iterrows():
        osmid = str(row.get("osmid", idx))
        name = _pick_name(row)
        geometry = row.geometry
        centroid = geometry.centroid
        lat, lon = centroid.y, centroid.x

        # determine category by first matching key/value
        category = None
        for cat_name, tags in build_category_tag_filters():
            for key, values in tags.items():
                val = row.get(key)
                values_in = set(map(str, val)) if isinstance(val, (list, tuple, set)) else ({str(val)} if val is not None else set())
                if any(v in values_in for v in values):
                    category = cat_name
                    break
            if category:
                break

        # Collect tags into strings
        tags_out: Dict[str, str] = {}
        for k, v in row.items():
            if k == "geometry":
                continue
            if isinstance(v, (list, tuple, set)):
                v_str = ",".join(map(str, v))
            else:
                v_str = str(v)
            tags_out[k] = v_str

        text_blob = " ".join([str(v).lower() for v in tags_out.values()])
        has_plaque = "plaque" in text_blob or (str(row.get("memorial:type")).lower() == "plaque")

        records.append({
            "osmid": osmid,
            "name": name,
            "lat": lat,
            "lon": lon,
            "category": category or "unknown",
            "tags": tags_out,
            "wikidata": row.get("wikidata"),
            "wikipedia": row.get("wikipedia"),
            "image": row.get("image"),
            "has_plaque": bool(has_plaque),
        })

    return pd.DataFrame.from_records(records)


def score_pois(df: pd.DataFrame) -> pd.DataFrame:
    def score_row(row: pd.Series) -> float:
        score = 0.0
        cat = str(row.get("category") or "").lower()
        tags: Dict[str, str] = row.get("tags", {}) or {}

        # Category weights
        score += {
            "historic": 6.0,
            "tourism": 4.0,
            "leisure": 2.0,
            "man_made": 2.0,
            "natural": 1.0,
        }.get(cat, 0.0)

        text = " ".join([str(v).lower() for v in tags.values()])
        boosts = [
            ("memorial", 4), ("monument", 4), ("plaque", 3), ("statue", 2), ("artwork", 2),
            ("viewpoint", 3), ("museum", 5), ("visitor", 2), ("heritage", 3), ("garden", 1), ("fountain", 1), ("bridge", 2)
        ]
        for token, w in boosts:
            if token in text:
                score += w

        penalties = ["toilet", "parking", "bench", "waste", "trash", "dog"]
        if any(p in text for p in penalties):
            score -= 3

        name = str(row.get("name") or "").lower()
        if any(k in name for k in ["memorial", "monument", "plaza", "statue", "garden", "museum", "bridge"]):
            score += 2

        if row.get("has_plaque"):
            score += 2

        if row.get("wikidata") or row.get("wikipedia"):
            score += 2

        return score

    df = df.copy()
    if df.empty:
        return df
    df["score"] = df.apply(score_row, axis=1)
    df = df[df["score"] > 0].reset_index(drop=True)
    return df


def cluster_and_select(df: pd.DataFrame, max_points: int = 15, max_total_miles: float = 3.0, distance_penalty: float = 1.0) -> pd.DataFrame:
    if df.empty:
        return df
    # Exclude unnamed POIs from consideration
    df = df[df["name"].astype(str).str.strip().str.lower() != "unnamed"].reset_index(drop=True)
    if df.empty:
        return df

    from math import radians, sin, cos, asin, sqrt

    def haversine_miles(lat1, lon1, lat2, lon2):
        R = 3958.8
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2) ** 2
        c = 2 * asin(sqrt(a))
        return R * c

    df_sorted = df.sort_values(["score"], ascending=False).reset_index(drop=True)

    # Greedy nearest-neighbor from highest-score seed
    selected: List[int] = [0]
    total_miles = 0.0

    while len(selected) < max_points and len(selected) < len(df_sorted):
        last = df_sorted.iloc[selected[-1]]
        best_idx = None
        best_rank = -1e9
        best_dist = 0.0
        for idx in range(len(df_sorted)):
            if idx in selected:
                continue
            cand = df_sorted.iloc[idx]
            d = haversine_miles(last.lat, last.lon, cand.lat, cand.lon)
            if total_miles + d > max_total_miles:
                continue
            rank = cand.score - (distance_penalty * d)  # balance quality and distance
            if rank > best_rank:
                best_rank = rank
                best_idx = idx
                best_dist = d
        if best_idx is None:
            break
        selected.append(best_idx)
        total_miles += best_dist

    return df_sorted.iloc[selected].reset_index(drop=True)


def export_csv(df: pd.DataFrame, park_name: str, output_path: str) -> None:
    df_out = df.copy()
    df_out.insert(0, "park", park_name)
    # Extract a few tag fields into flat columns
    for key in ["operator", "website", "addr:city", "start_date", "opening_hours"]:
        df_out[key] = df_out["tags"].apply(lambda t: t.get(key) if isinstance(t, dict) else None)
    cols = [
        "park", "name", "lat", "lon", "category", "score", "has_plaque",
        "wikidata", "wikipedia", "image", "operator", "website", "start_date", "opening_hours", "osmid",
        "google_place_id", "google_name", "google_rating", "google_ratings_total"
    ]
    df_out = df_out[[c for c in cols if c in df_out.columns]]
    df_out.to_csv(output_path, index=False)


def _ensure_cache_dir() -> str:
    base_dir = os.path.join(os.path.dirname(__file__), "cache", "google_places")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _normalize_name(name: str) -> List[str]:
    s = re.sub(r"[^\w\s]", " ", str(name).lower())
    tokens = [t for t in s.split() if t]
    return tokens


def _token_jaccard(a: str, b: str) -> float:
    ta = set(_normalize_name(a))
    tb = set(_normalize_name(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, asin, sqrt
    R_km = 6371.0088
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2) ** 2
    c = 2 * asin(sqrt(a))
    return R_km * c * 1000.0


def _cache_key(name: str, lat: float, lon: float, radius_m: int) -> str:
    sig = f"{name}|{lat:.5f}|{lon:.5f}|{radius_m}"
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()


def _read_cache(cache_dir: str, key: str) -> Optional[dict]:
    path = os.path.join(cache_dir, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(cache_dir: str, key: str, data: dict) -> None:
    path = os.path.join(cache_dir, f"{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def _google_text_search(name: str, lat: float, lon: float, radius_meters: int, api_key: str, timeout_sec: int = 10) -> Optional[dict]:
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "key": api_key,
        "query": name,
        "location": f"{lat},{lon}",
        "radius": str(radius_meters),
    }
    try:
        resp = requests.get(url, params=params, timeout=timeout_sec)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("status") not in {"OK", "ZERO_RESULTS"}:
            return None
        results = data.get("results", [])
        if not results:
            return None
        # Rank results by name similarity and proximity
        ranked = []
        for r in results:
            r_name = r.get("name") or ""
            r_loc = r.get("geometry", {}).get("location", {})
            r_lat = r_loc.get("lat")
            r_lng = r_loc.get("lng")
            if r_lat is None or r_lng is None:
                continue
            dist_m = _haversine_meters(lat, lon, float(r_lat), float(r_lng))
            name_sim = _token_jaccard(name, r_name)
            rank = name_sim * 2.0 - (dist_m / 1000.0)
            ranked.append((rank, name_sim, -dist_m, r))
        if not ranked:
            return None
        ranked.sort(reverse=True)
        best = ranked[0][3]
        return best
    except Exception:
        return None


def _rating_boost(rating: Optional[float], ratings_total: Optional[int]) -> float:
    if rating is None:
        return 0.0
    try:
        r = float(rating)
    except Exception:
        return 0.0
    boost = 0.0
    if r >= 4.8:
        boost = 2.0
    elif r >= 4.5:
        boost = 1.5
    elif r >= 4.3:
        boost = 1.0
    elif r >= 4.0:
        boost = 0.5
    # Lightly weight by review count (cap at +1 extra)
    try:
        n = int(ratings_total or 0)
        if n >= 50:
            boost += min(0.5, n / 1000.0)
    except Exception:
        pass
    return boost


def enrich_with_google_places(df: pd.DataFrame, api_key: str, radius_meters: int = 150, sleep_between_s: float = 0.1) -> pd.DataFrame:
    if df.empty:
        return df
    cache_dir = _ensure_cache_dir()
    df = df.copy()
    # Prepare columns
    for c in ["google_place_id", "google_name", "google_rating", "google_ratings_total"]:
        if c not in df.columns:
            df[c] = None
    if "score" not in df.columns:
        df["score"] = 0.0
    for idx, row in df.iterrows():
        name = str(row.get("name") or "").strip()
        if not name or name.lower() == "unnamed":
            continue
        lat = float(row.get("lat"))
        lon = float(row.get("lon"))
        cache_key = _cache_key(name, lat, lon, radius_meters)
        cached = _read_cache(cache_dir, cache_key)
        place = cached
        if place is None:
            place = _google_text_search(name, lat, lon, radius_meters, api_key)
            if place is not None:
                _write_cache(cache_dir, cache_key, place)
            # be polite
            if sleep_between_s > 0:
                time.sleep(sleep_between_s)
        if not place:
            continue
        place_id = place.get("place_id")
        g_name = place.get("name")
        rating = place.get("rating")
        ratings_total = place.get("user_ratings_total")
        df.at[idx, "google_place_id"] = place_id
        df.at[idx, "google_name"] = g_name
        df.at[idx, "google_rating"] = rating
        df.at[idx, "google_ratings_total"] = ratings_total
        df.at[idx, "score"] = float(df.at[idx, "score"]) + _rating_boost(rating, ratings_total)
    return df


def process_park(park: Optional[str], city: Optional[str], output_csv: str, max_points: int, max_total_miles: float, min_park_area_m2: float, enable_google_places: bool = False, google_api_key: Optional[str] = None, google_radius_meters: int = 150, distance_penalty: float = 1.0) -> Tuple[str, pd.DataFrame]:
    if park:
        park_name, polygon = get_park_polygon_by_name(park)
    elif city:
        park_name, polygon = get_largest_park_in_city(city, min_area_m2=min_park_area_m2)
    else:
        raise ValueError("Either --park or --city must be provided")

    pois = query_pois_within_polygon(polygon)
    scored = score_pois(pois)
    if enable_google_places:
        if not google_api_key:
            # allow env var fallback
            google_api_key = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Google Places enrichment enabled but no API key provided (use --google-api-key or set GOOGLE_MAPS_API_KEY)")
        scored = enrich_with_google_places(scored, api_key=google_api_key, radius_meters=google_radius_meters)
    selected = cluster_and_select(scored, max_points=max_points, max_total_miles=max_total_miles, distance_penalty=distance_penalty)
    export_csv(selected, park_name=park_name, output_path=output_csv)
    return park_name, selected


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CSV of interesting POIs for a park using OSMnx/Overpass")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--park", type=str, help="Specific park name or query string")
    group.add_argument("--city", type=str, help="City query; will pick the largest park")
    parser.add_argument("--output", type=str, default="park_pois.csv", help="Output CSV path")
    parser.add_argument("--max-points", type=int, default=15, help="Max number of POIs to select")
    parser.add_argument("--max-miles", type=float, default=3.0, help="Max total walking miles")
    parser.add_argument("--min-park-area-m2", type=float, default=300_000.0, help="Minimum area for candidate parks when using --city")
    parser.add_argument("--distance-penalty", type=float, default=5.0, help="Multiplier on distance in rank = score − (penalty × miles)")
    parser.add_argument("--enable-google-places", action="store_true", help="Enrich POIs using Google Places (adds ratings and IDs)")
    parser.add_argument("--google-api-key", type=str, default=None, help="Google Maps/Places API key (or set GOOGLE_MAPS_API_KEY)")
    parser.add_argument("--google-radius-meters", type=int, default=150, help="Search radius for Places matching (meters)")
    # Sensible default: largest park in Pittsburgh
    parser.set_defaults(city="Pittsburgh, Pennsylvania", park=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        park_name, df = process_park(
            park=args.park,
            city=args.city,
            output_csv=args.output,
            max_points=args.max_points,
            max_total_miles=args.max_miles,
            min_park_area_m2=args.min_park_area_m2,
            enable_google_places=args.enable_google_places,
            google_api_key=args.google_api_key,
            google_radius_meters=args.google_radius_meters,
            distance_penalty=args.distance_penalty,
        )
        print(f"Selected {len(df)} POIs for {park_name}. Wrote: {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
