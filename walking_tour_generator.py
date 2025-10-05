#!/usr/bin/env python3

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

# Reuse POI pipeline and utilities from the existing project to avoid duplication
from park_game_generator import (
    query_pois_within_polygon,
    score_pois,
    enrich_with_google_places,
)


# Configure OSMnx settings similarly to the existing generator for stability across runs.
ox.settings.use_cache = True
ox.settings.log_console = False
try:
    ox.settings.requests_timeout = 180
except Exception:
    try:
        ox.settings.timeout = 180
    except Exception:
        pass
ox.settings.overpass_rate_limit = True


# ---------------------------
# Small data structures
# ---------------------------


@dataclass
class GeoPoint:
    """Simple geographic point in WGS84.

    Using a tiny dataclass improves clarity when passing around coordinates,
    and supports adding display labels alongside lat/lon.
    """

    lat: float
    lon: float
    label: str


@dataclass
class DirectionStep:
    """One human-readable instruction with distance metadata."""

    text: str
    distance_m: float
    start_m: float
    end_m: float


@dataclass
class PoiCallout:
    """POI mention bound to a position along the route."""

    name: str
    description: str
    along_m: float
    side: Optional[str]


# ---------------------------
# Argument parsing
# ---------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for walking tour generation.

    Keep options compact and mirror naming from the existing park generator where useful.
    """

    parser = argparse.ArgumentParser(description="Generate walking directions with interleaved POI tour text")
    parser.add_argument("--start", required=True, help="Start location address or 'lat,lon'")
    parser.add_argument("--end", required=True, help="End location address or 'lat,lon'")
    parser.add_argument("--output", default="walking_tour.txt", help="Output .txt path")
    parser.add_argument("--buffer-meters", type=int, default=100, help="Half-width of corridor for POIs")
    parser.add_argument("--max-pois", type=int, default=15, help="Max number of POIs to include in the tour")
    parser.add_argument("--min-poi-score", type=float, default=1.0, help="Minimum score to consider a POI")
    parser.add_argument(
        "--graph-distance-m",
        type=int,
        default=None,
        help="Distance (meters) around midpoint to load walk network; default auto",
    )
    parser.add_argument("--enable-google-places", action="store_true", help="Enrich POIs with Google ratings")
    parser.add_argument("--google-api-key", default=None, help="Google API key (or set GOOGLE_MAPS_API_KEY)")
    parser.add_argument("--google-radius-meters", type=int, default=150, help="Search radius for Places matching (m)")
    return parser.parse_args(argv)


# ---------------------------
# Geocoding and graph loading
# ---------------------------


def _try_parse_latlon(text: str) -> Optional[Tuple[float, float]]:
    """Parse 'lat,lon' into floats; return None if not in that form.

    Robustly handling coordinates avoids unnecessary geocoding requests and speeds up runs.
    """

    try:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 2:
            return None
        lat = float(parts[0])
        lon = float(parts[1])
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return None
        return lat, lon
    except Exception:
        return None


def geocode_point(raw: str) -> GeoPoint:
    """Convert a user-supplied string into a WGS84 point.

    Accepts direct coordinates or any geocodable description OSMnx supports.
    """

    parsed = _try_parse_latlon(raw)
    if parsed is not None:
        lat, lon = parsed
        return GeoPoint(lat=lat, lon=lon, label=f"{lat:.5f},{lon:.5f}")
    gdf = ox.geocode_to_gdf(raw)
    if gdf.empty:
        raise ValueError(f"Could not geocode location: {raw}")
    geom = gdf.iloc[0].geometry
    centroid = geom.centroid
    display = str(gdf.iloc[0].get("display_name") or raw)
    return GeoPoint(lat=float(centroid.y), lon=float(centroid.x), label=display)


def _auto_graph_distance_m(start: GeoPoint, end: GeoPoint) -> int:
    """Choose a graph radius that comfortably covers start→end with slack.

    We use great-circle distance plus a margin so the route has enough network coverage
    even if the road network meanders.
    """

    # Simple haversine; avoids importing more utils here to keep the module cohesive.
    def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r_km = 6371.0088
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return r_km * c * 1000.0

    straight_m = haversine_m(start.lat, start.lon, end.lat, end.lon)
    margin = max(1500.0, straight_m * 0.6)
    # Cap to a reasonable upper bound to reduce over-downloading large graphs by accident.
    return int(min(15000.0, straight_m + margin))


def load_walk_graph(start: GeoPoint, end: GeoPoint, distance_m: Optional[int]) -> nx.MultiDiGraph:
    """Load a walking network around the midpoint covering the desired corridor."""

    if distance_m is None:
        distance_m = _auto_graph_distance_m(start, end)
    mid_lat = (start.lat + end.lat) / 2.0
    mid_lon = (start.lon + end.lon) / 2.0
    G = ox.graph_from_point((mid_lat, mid_lon), dist=distance_m, network_type="walk", simplify=True)
    # Bearings help produce human-readable turn instructions
    _add_edge_bearings_safe(G)
    return G


def _compute_initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing from point 1 to 2 in degrees (0..360).

    We use spherical approximation which is sufficient for turn phrasing.
    """

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    brng = (math.degrees(math.atan2(x, y)) + 360.0) % 360.0
    return brng


def _add_edge_bearings_safe(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Add 'bearing' to edges using whichever OSMnx API is available; fallback to manual.

    OSMnx changed bearing helpers across versions. This wrapper allows the same
    code to run on 1.x without pinning a specific minor version.
    """

    try:
        # Preferred modern API
        if hasattr(ox, "bearing") and hasattr(ox.bearing, "add_edge_bearings"):
            return ox.bearing.add_edge_bearings(G)
    except Exception:
        pass
    try:
        # Older API sometimes exposes via utils_graph
        if hasattr(ox, "utils_graph") and hasattr(ox.utils_graph, "add_edge_bearings"):
            return ox.utils_graph.add_edge_bearings(G)
    except Exception:
        pass
    try:
        # Very old convenience function
        if hasattr(ox, "add_edge_bearings"):
            return ox.add_edge_bearings(G)
    except Exception:
        pass

    # Manual fallback: compute and set once per (u,v,k)
    for u, v, k, data in G.edges(keys=True, data=True):
        if "geometry" in data and data["geometry"] and len(data["geometry"].coords) >= 2:
            (x1, y1) = data["geometry"].coords[0]
            (x2, y2) = data["geometry"].coords[-1]
            lat1, lon1 = y1, x1
            lat2, lon2 = y2, x2
        else:
            lat1 = float(G.nodes[u]["y"])  # lat
            lon1 = float(G.nodes[u]["x"])  # lon
            lat2 = float(G.nodes[v]["y"])  # lat
            lon2 = float(G.nodes[v]["x"])  # lon
        data["bearing"] = _compute_initial_bearing(lat1, lon1, lat2, lon2)
    return G


# ---------------------------
# Routing and directions
# ---------------------------


def compute_route(G: nx.MultiDiGraph, start: GeoPoint, end: GeoPoint) -> List[int]:
    """Compute shortest path by length between nearest graph nodes."""

    start_node = ox.distance.nearest_nodes(G, X=[start.lon], Y=[start.lat])[0]
    end_node = ox.distance.nearest_nodes(G, X=[end.lon], Y=[end.lat])[0]
    return ox.shortest_path(G, start_node, end_node, weight="length")


def _edge_name(data: dict) -> str:
    """Pick a readable street/path name, falling back to neutral phrasing."""

    name = data.get("name")
    if isinstance(name, list) and name:
        name = name[0]
    if isinstance(name, str) and name.strip():
        return name
    # Trails and footpaths are often unnamed; keep output friendly.
    return "path"


def _bearing_delta(b1: Optional[float], b2: Optional[float]) -> Optional[float]:
    """Smallest signed delta (degrees) from b1→b2 in [-180, 180]."""

    if b1 is None or b2 is None:
        return None
    d = (b2 - b1 + 180.0) % 360.0 - 180.0
    return d


def _turn_phrase(delta: Optional[float]) -> Optional[str]:
    """Translate a bearing change into a concise instruction word.

    We bias toward fewer strong turn words to keep directions readable.
    """

    if delta is None:
        return None
    ad = abs(delta)
    if ad < 15:
        return None  # continue
    if ad < 35:
        return "slight left" if delta < 0 else "slight right"
    if ad < 100:
        return "left" if delta < 0 else "right"
    if ad < 160:
        return "sharp left" if delta < 0 else "sharp right"
    return "u-turn"


def _cardinal_from_bearing(b: Optional[float]) -> Optional[str]:
    if b is None:
        return None
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((b % 360) + 22.5) // 45) % 8
    return dirs[idx]


def _format_distance_m(m: float) -> str:
    """Readable walking distance using miles for longer segments and meters for very short ones."""

    mi = m / 1609.344
    if mi >= 0.25:
        return f"{mi:.2f} mi"
    if m >= 100:
        return f"{int(round(m, -1))} m"
    return f"{int(round(m))} m"


def build_directions(
    G: nx.MultiDiGraph,
    route: Sequence[int],
) -> Tuple[List[DirectionStep], LineString, float]:
    """Convert a node route into readable steps.

    Returns the steps, a LineString for the full geometry, and total length in meters.
    We merge consecutive edges with similar names and bearings to reduce verbosity.
    """

    if not route or len(route) < 2:
        return [], LineString([]), 0.0

    edges = ox.utils_graph.get_route_edge_attributes(G, route, attribute=None)
    coords: List[Tuple[float, float]] = []
    steps: List[DirectionStep] = []

    # Build full coordinate chain and per-edge info
    total_m = 0.0
    current_name: Optional[str] = None
    current_bearing: Optional[float] = None
    seg_start_m = 0.0
    seg_dist_m = 0.0
    step_start_m = 0.0

    for i, data in enumerate(edges):
        length_m = float(data.get("length") or 0.0)
        total_m += length_m
        seg_dist_m += length_m
        if "geometry" in data and data["geometry"]:
            geom: LineString = data["geometry"]
            if not coords:
                coords.extend(list(geom.coords))
            else:
                coords.extend(list(geom.coords)[1:])
        else:
            u = data.get("u")
            v = data.get("v")
            if u in G.nodes and v in G.nodes:
                coords_u = (G.nodes[u]["x"], G.nodes[u]["y"])  # lon, lat
                coords_v = (G.nodes[v]["x"], G.nodes[v]["y"])  # lon, lat
                if not coords:
                    coords.append(coords_u)
                coords.append(coords_v)

        name = _edge_name(data)
        bearing = data.get("bearing")

        # Decide if we should start a new step based on name/bearing change
        if current_name is None:
            current_name = name
            current_bearing = bearing
            step_start_m = seg_start_m
            continue

        delta = _bearing_delta(current_bearing, bearing)
        turn = _turn_phrase(delta)
        should_split = (name != current_name) or (turn is not None)
        if should_split:
            cardinal = _cardinal_from_bearing(current_bearing)
            heading = f"Head {cardinal} on {current_name}" if not steps else f"Turn {turn} onto {current_name if name != current_name else current_name}"
            steps.append(
                DirectionStep(
                    text=f"{heading} for {_format_distance_m(seg_dist_m)}",
                    distance_m=seg_dist_m,
                    start_m=step_start_m,
                    end_m=step_start_m + seg_dist_m,
                )
            )
            # Start a new segment
            current_name = name
            current_bearing = bearing
            step_start_m = step_start_m + seg_dist_m
            seg_dist_m = 0.0

    # Flush tail
    if seg_dist_m > 0.0 and current_name is not None:
        cardinal = _cardinal_from_bearing(current_bearing)
        heading = f"Head {cardinal} on {current_name}" if not steps else f"Continue on {current_name}"
        steps.append(
            DirectionStep(
                text=f"{heading} for {_format_distance_m(seg_dist_m)}",
                distance_m=seg_dist_m,
                start_m=step_start_m,
                end_m=step_start_m + seg_dist_m,
            )
        )

    line = LineString([(lon, lat) for lon, lat in coords])
    return steps, line, total_m


# ---------------------------
# POIs along the corridor
# ---------------------------


def _to_3857(geom) -> gpd.GeoSeries:
    """Project a single geometry to EPSG:3857 using GeoPandas for correctness."""

    return gpd.GeoSeries([geom], crs=4326).to_crs(3857)


def _to_4326(series: gpd.GeoSeries):
    return series.to_crs(4326)


def build_corridor_polygon(route_wgs84: LineString, buffer_meters: int) -> Polygon:
    """Create a buffered corridor polygon around the route centerline.

    Buffering in a metric CRS avoids distortions and makes width predictable.
    """

    if route_wgs84.is_empty:
        return Polygon()
    route_m = _to_3857(route_wgs84)
    poly_m = route_m.buffer(buffer_meters)
    return _to_4326(poly_m).iloc[0]


def fetch_scored_pois_in_corridor(
    corridor: Polygon,
    min_score: float,
    enable_google_places: bool,
    google_api_key: Optional[str],
    google_radius_m: int,
) -> pd.DataFrame:
    """Query and score POIs inside the corridor polygon, with optional Google enrichment."""

    df = query_pois_within_polygon(corridor)
    df = score_pois(df)
    # Exclude unnamed POIs to keep the narrative meaningful
    if not df.empty and "name" in df.columns:
        names = df["name"].astype(str).str.strip()
        df = df[(names.str.len() > 0) & (names.str.lower() != "unnamed")].reset_index(drop=True)
    if min_score is not None and not df.empty:
        df = df[df["score"] >= float(min_score)].reset_index(drop=True)
    if enable_google_places and not df.empty:
        if not google_api_key:
            google_api_key = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            df = enrich_with_google_places(df, api_key=google_api_key, radius_meters=google_radius_m)
    return df


def _line_along_positions_m(line_m: LineString, points_m: Iterable[Point]) -> List[float]:
    """Compute along-route positions (meters from start) for many points in metric CRS."""

    return [float(line_m.project(p)) for p in points_m]


def _perpendicular_distance_m(line_m: LineString, point_m: Point) -> float:
    """Shortest distance from the line to a point (meters)."""

    return float(line_m.distance(point_m))


def _segment_side(line_m: LineString, point_m: Point, along_m: float) -> Optional[str]:
    """Infer left/right side at a local segment near a given along-route position.

    We compute the nearest segment by sampling the coordinate sequence around the projected point
    and use a 2D cross-product sign to classify left vs right. This is approximate but robust.
    """

    if len(line_m.coords) < 2:
        return None
    coords = list(line_m.coords)
    # Find nearest vertex index along the chain to the projected point
    min_d = float("inf")
    min_i = 0
    px, py = point_m.x, point_m.y
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        # Segment midpoint heuristic keeps it simple
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        d = (mx - px) ** 2 + (my - py) ** 2
        if d < min_d:
            min_d = d
            min_i = i
    x1, y1 = coords[min_i]
    x2, y2 = coords[min_i + 1]
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    cross = vx * wy - vy * wx
    if abs(cross) < 1e-9:
        return None
    return "left" if cross > 0 else "right"


def select_and_order_pois(
    df: pd.DataFrame,
    route_wgs84: LineString,
    max_pois: int,
) -> Tuple[pd.DataFrame, List[PoiCallout]]:
    """Choose top POIs near the path and order them along the walk.

    We balance intrinsic score with perpendicular distance to keep mentions relevant.
    """

    if df.empty or route_wgs84.is_empty:
        return df.head(0), []

    line_m = _to_3857(route_wgs84).iloc[0]
    pts = [Point(lon, lat) for lon, lat in zip(df["lon"].astype(float), df["lat"].astype(float))]
    pts_m = [gpd.GeoSeries([p], crs=4326).to_crs(3857).iloc[0] for p in pts]
    alongs = _line_along_positions_m(line_m, pts_m)
    dists = [_perpendicular_distance_m(line_m, p) for p in pts_m]

    df = df.copy()
    df["along_m"] = alongs
    df["perp_m"] = dists

    # Composite rank: prioritize score and proximity; penalize far-off POIs
    df["rank"] = df["score"] * 2.0 - (df["perp_m"] / 50.0)
    df = df.sort_values(["rank", "score"], ascending=[False, False]).head(max_pois).copy()
    df = df.sort_values("along_m", ascending=True).reset_index(drop=True)

    callouts: List[PoiCallout] = []
    for i, row in df.iterrows():
        p_m = pts_m[i]
        side = _segment_side(line_m, p_m, along_m=row["along_m"])  # approximate
        desc_parts: List[str] = []
        if str(row.get("category") or "").strip():
            desc_parts.append(str(row.get("category")))
        if bool(row.get("has_plaque")):
            desc_parts.append("plaque")
        # Add Google rating if present
        rating = row.get("google_rating")
        ratings_total = row.get("google_ratings_total")
        if pd.notna(rating):
            if pd.notna(ratings_total):
                desc_parts.append(f"rated {float(rating):.1f} ({int(ratings_total)} reviews)")
            else:
                desc_parts.append(f"rated {float(rating):.1f}")
        description = ", ".join(desc_parts) if desc_parts else "point of interest"
        callouts.append(
            PoiCallout(
                name=str(row.get("name") or "Unnamed"),
                description=description,
                along_m=float(row["along_m"]),
                side=side,
            )
        )

    return df, callouts


def weave_pois_into_steps(steps: List[DirectionStep], callouts: List[PoiCallout]) -> List[str]:
    """Produce final narrative lines: steps plus nearby POI callouts inline.

    We align POIs to the step whose distance range contains the POI position; if none, attach to the next step.
    """

    lines: List[str] = []
    step_iter = iter(steps)
    current = next(step_iter, None)
    callout_idx = 0
    step_num = 1
    while current is not None:
        lines.append(f"{step_num}. {current.text}")
        # Emit all POIs that fall within this step's span
        while callout_idx < len(callouts):
            c = callouts[callout_idx]
            if c.along_m <= current.end_m + 1e-6:
                side = f" on your {c.side}" if c.side else ""
                lines.append(f"   - You'll pass{side} {c.name} ({c.description}).")
                callout_idx += 1
            else:
                break
        current = next(step_iter, None)
        step_num += 1
    # Any remaining callouts go at the end; this is rare but keeps content.
    while callout_idx < len(callouts):
        c = callouts[callout_idx]
        side = f" on your {c.side}" if c.side else ""
        lines.append(f"   - Ahead{side}: {c.name} ({c.description}).")
        callout_idx += 1
    return lines


# ---------------------------
# Output formatting
# ---------------------------


def _estimate_walk_time_minutes(total_m: float, speed_kmh: float = 4.5) -> int:
    """Convert distance to minutes using a moderate walking pace."""

    if total_m <= 0:
        return 0
    km = total_m / 1000.0
    hours = km / speed_kmh
    return int(round(hours * 60.0))


def format_tour_text(
    start: GeoPoint,
    end: GeoPoint,
    steps: List[DirectionStep],
    callout_lines: List[str],
    total_m: float,
) -> str:
    """Assemble the final plain-text tour document."""

    header = [
        f"Walking tour from: {start.label}",
        f"Destination: {end.label}",
        f"Total distance: {_format_distance_m(total_m)}",
        f"Estimated time: {_estimate_walk_time_minutes(total_m)} min",
        "",
        "Directions:",
    ]
    body = []
    for line in callout_lines:
        body.append(line)
    footer = ["", "Enjoy your walk!"]
    return "\n".join(header + body + footer)


# ---------------------------
# Main flow
# ---------------------------


def run(argv: Optional[Sequence[str]] = None) -> int:
    """Entrypoint for generating the tour.

    We separate execution into small, testable functions for readability and maintainability.
    """

    args = parse_args(argv)
    try:
        start = geocode_point(args.start)
        end = geocode_point(args.end)

        G = load_walk_graph(start, end, args.graph_distance_m)
        route = compute_route(G, start, end)
        steps, route_line, total_m = build_directions(G, route)

        corridor = build_corridor_polygon(route_line, buffer_meters=int(args.buffer_meters))
        poi_df = fetch_scored_pois_in_corridor(
            corridor,
            min_score=float(args.min_poi_score),
            enable_google_places=bool(args.enable_google_places),
            google_api_key=args.google_api_key,
            google_radius_m=int(args.google_radius_meters),
        )
        _, callouts = select_and_order_pois(poi_df, route_line, max_pois=int(args.max_pois))
        narrative_lines = weave_pois_into_steps(steps, callouts)

        text = format_tour_text(start, end, steps, narrative_lines, total_m)
        out_path = os.path.abspath(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote walking tour to: {out_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
