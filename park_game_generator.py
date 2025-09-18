#!/usr/bin/env python3

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

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


def cluster_and_select(df: pd.DataFrame, max_points: int = 10, max_total_miles: float = 2.0) -> pd.DataFrame:
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
            rank = cand.score - d  # balance quality and distance
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
    df_out["extra_info"] = df_out["tags"].apply(lambda t: json.dumps(t) if isinstance(t, dict) else str(t))
    cols = [
        "park", "name", "lat", "lon", "category", "score", "has_plaque",
        "wikidata", "wikipedia", "image", "operator", "website", "start_date", "opening_hours", "extra_info", "osmid"
    ]
    df_out = df_out[[c for c in cols if c in df_out.columns]]
    df_out.to_csv(output_path, index=False)


def process_park(park: Optional[str], city: Optional[str], output_csv: str, max_points: int, max_total_miles: float, min_park_area_m2: float) -> Tuple[str, pd.DataFrame]:
    if park:
        park_name, polygon = get_park_polygon_by_name(park)
    elif city:
        park_name, polygon = get_largest_park_in_city(city, min_area_m2=min_park_area_m2)
    else:
        raise ValueError("Either --park or --city must be provided")

    pois = query_pois_within_polygon(polygon)
    scored = score_pois(pois)
    selected = cluster_and_select(scored, max_points=max_points, max_total_miles=max_total_miles)
    export_csv(selected, park_name=park_name, output_path=output_csv)
    return park_name, selected


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CSV of interesting POIs for a park using OSMnx/Overpass")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--park", type=str, help="Specific park name or query string")
    group.add_argument("--city", type=str, help="City query; will pick the largest park")
    parser.add_argument("--output", type=str, default="park_pois.csv", help="Output CSV path")
    parser.add_argument("--max-points", type=int, default=10, help="Max number of POIs to select")
    parser.add_argument("--max-miles", type=float, default=2.0, help="Max total walking miles")
    parser.add_argument("--min-park-area-m2", type=float, default=300_000.0, help="Minimum area for candidate parks when using --city")
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
        )
        print(f"Selected {len(df)} POIs for {park_name}. Wrote: {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
