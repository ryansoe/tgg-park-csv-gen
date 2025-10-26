#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import transform

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
    
    # Label customization
    parser.add_argument("--start-label", default=None, help="Override display name for start location")
    parser.add_argument("--end-label", default=None, help="Override display name for end location")
    
    # Distance display
    parser.add_argument("--hide-step-distances", action="store_true", help="Omit per-step distance text")
    
    # POI callout tuning
    parser.add_argument("--max-callouts-per-step", type=int, default=1, help="Max POI callouts per step")
    parser.add_argument("--callout-style", choices=["minimal", "descriptive"], default="minimal", help="Callout verbosity")
    parser.add_argument("--preferred-categories", default=None, help="CSV list of preferred POI categories")
    parser.add_argument("--blocked-categories", default=None, help="CSV list of blocked POI categories")
    
    # LLM post-processing
    parser.add_argument("--llm-enabled", action="store_true", help="Enable LLM post-processing of directions")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="LLM temperature (0-1)")
    parser.add_argument("--llm-complexity", choices=["simple", "medium", "complex"], default="simple", help="LLM complexity level: simple (concise), medium (moderate detail), complex (rich narrative)")
    parser.add_argument("--llm-max-steps", type=int, default=10, help="Max steps for LLM to produce")
    parser.add_argument("--llm-include-distances", action="store_true", help="Include per-step distances in LLM output")
    
    # Unit system
    parser.add_argument("--units", choices=["metric", "imperial"], default="imperial", help="Unit system: metric (m/km) or imperial (ft/mi)")
    
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


def _derive_label_from_nearby_features(lat: float, lon: float, radius_m: int = 80) -> Optional[str]:
    """Try to find a nearby named building or amenity using OSM features.
    
    Returns a human-friendly label or None if nothing suitable is found.
    """
    try:
        # Query nearby features that might have useful names
        tags = {
            "building": True,
            "amenity": True,
            "tourism": True,
            "leisure": True,
            "historic": True,
        }
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius_m)
        if gdf.empty:
            return None
        
        # Filter to entries with names
        if "name" not in gdf.columns:
            return None
        named = gdf[gdf["name"].notna() & (gdf["name"].astype(str).str.strip() != "")].copy()
        if named.empty:
            return None
        
        # Compute distances and pick closest
        from shapely.geometry import Point as ShapelyPoint
        pt = ShapelyPoint(lon, lat)
        named["dist"] = named.geometry.apply(lambda g: g.distance(pt))
        closest = named.sort_values("dist").iloc[0]
        return str(closest["name"]).strip()
    except Exception:
        return None


def geocode_point(raw: str, label_override: Optional[str] = None) -> GeoPoint:
    """Convert a user-supplied string into a WGS84 point.

    Accepts direct coordinates or any geocodable description OSMnx supports.
    If label_override is provided, it will be used as the display label.
    For lat/lon inputs, attempts to derive a nearby named feature as the label.
    """

    parsed = _try_parse_latlon(raw)
    if parsed is not None:
        lat, lon = parsed
        if label_override:
            label = label_override
        else:
            # Try to find a nearby named feature
            derived = _derive_label_from_nearby_features(lat, lon)
            label = derived if derived else f"{lat:.5f},{lon:.5f}"
        return GeoPoint(lat=lat, lon=lon, label=label)
    
    gdf = ox.geocode_to_gdf(raw)
    if gdf.empty:
        raise ValueError(f"Could not geocode location: {raw}")
    geom = gdf.iloc[0].geometry
    centroid = geom.centroid
    if label_override:
        display = label_override
    else:
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


def _edge_name(data: dict) -> Tuple[str, str]:
    """Pick a readable street/path name, falling back to neutral phrasing.
    
    Returns (name, facility_type) where facility_type is used when name is generic.
    facility_type can be "trail", "footpath", or "" (omit).
    """

    name = data.get("name")
    if isinstance(name, list) and name:
        name = name[0]
    if isinstance(name, str) and name.strip():
        return name, ""
    
    # No explicit name; check highway tag to pick a friendly facility type
    highway = data.get("highway", "")
    if isinstance(highway, list) and highway:
        highway = highway[0]
    highway = str(highway).lower()
    
    if "trail" in highway or highway in ["path", "track"]:
        return "path", "trail"
    if "footway" in highway or highway == "pedestrian":
        return "path", "footpath"
    
    # Generic fallback
    return "path", ""


def _bearing_delta(b1: Optional[float], b2: Optional[float]) -> Optional[float]:
    """Smallest signed delta (degrees) from b1→b2 in [-180, 180]."""

    if b1 is None or b2 is None:
        return None
    d = (b2 - b1 + 180.0) % 360.0 - 180.0
    return d


def _turn_phrase(delta: Optional[float]) -> Optional[str]:
    """Translate a bearing change into a concise instruction word.

    We bias toward fewer strong turn words to keep directions readable.
    Updated thresholds: <15° = continue, <30° = slight, 30-90° = turn, 90-160° = sharp, ≥160° = u-turn.
    """

    if delta is None:
        return None
    ad = abs(delta)
    if ad < 15:
        return None  # continue - no new step
    if ad < 30:
        return "slight left" if delta < 0 else "slight right"
    if ad < 90:
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


def _format_distance_m(m: float, units: str = "imperial") -> str:
    """Readable walking distance in metric (m/km) or imperial (ft/mi) units.
    
    Args:
        m: Distance in meters
        units: "metric" or "imperial"
    
    Returns:
        Formatted distance string
    """
    if units == "metric":
        # Metric: use km for longer distances, m for shorter
        km = m / 1000.0
        if km >= 0.5:
            return f"{km:.2f} km"
        if m >= 100:
            return f"{int(round(m, -1))} m"
        return f"{int(round(m))} m"
    else:
        # Imperial: use miles for longer distances, feet for shorter
        ft = m * 3.28084
        mi = m / 1609.344
        if mi >= 0.25:
            return f"{mi:.2f} mi"
        if ft >= 100:
            return f"{int(round(ft, -1))} ft"
        return f"{int(round(ft))} ft"


def _detect_start_building(lat: float, lon: float) -> Optional[str]:
    """Check if the start point is inside a named building polygon.
    
    Returns the building name if found, else None.
    """
    try:
        from shapely.geometry import Point as ShapelyPoint
        pt = ShapelyPoint(lon, lat)
        
        # Query buildings within a small radius
        tags = {"building": True}
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=50)
        if gdf.empty or "name" not in gdf.columns:
            return None
        
        # Filter to named buildings that contain the point
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom and geom.contains(pt):
                name = row.get("name")
                if name and str(name).strip():
                    return str(name).strip()
        return None
    except Exception:
        return None


def _simplify_linestring(line: LineString, tolerance_m: float = 3.0) -> LineString:
    """Simplify a LineString in WGS84 using a metric tolerance.
    
    Projects to 3857, simplifies, then back to 4326.
    """
    if line.is_empty or len(line.coords) < 3:
        return line
    try:
        # Project to metric CRS
        gs = gpd.GeoSeries([line], crs=4326).to_crs(3857)
        simplified_m = gs.simplify(tolerance_m, preserve_topology=True)
        back = simplified_m.to_crs(4326)
        return back.iloc[0]
    except Exception:
        return line


def build_directions(
    G: nx.MultiDiGraph,
    route: Sequence[int],
    start_point: GeoPoint,
    hide_step_distances: bool = False,
    units: str = "imperial",
) -> Tuple[List[DirectionStep], LineString, float, Optional[str], Optional[str]]:
    """Convert a node route into readable steps with improved phrasing and merging.

    Returns:
        steps: list of DirectionStep
        line: LineString geometry for the route
        total_m: total distance in meters
        pre_step_building: optional "Walk out of <Building>" text
        pre_step_connector: optional "Walk to <Street>" text
    """

    if not route or len(route) < 2:
        return [], LineString([]), 0.0, None, None

    edges = ox.utils_graph.get_route_edge_attributes(G, route, attribute=None)
    coords: List[Tuple[float, float]] = []

    # Build full coordinate chain and collect edge info
    @dataclass
    class EdgeInfo:
        name: str
        facility_type: str
        bearing: Optional[float]
        length_m: float
        data: dict

    edge_infos: List[EdgeInfo] = []
    total_m = 0.0

    for i, data in enumerate(edges):
        length_m = float(data.get("length") or 0.0)
        total_m += length_m
        
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

        name, facility_type = _edge_name(data)
        bearing = data.get("bearing")
        edge_infos.append(EdgeInfo(name=name, facility_type=facility_type, bearing=bearing, length_m=length_m, data=data))

    line = LineString([(lon, lat) for lon, lat in coords])
    
    # Simplify geometry to reduce micro-turns
    line = _simplify_linestring(line, tolerance_m=3.0)

    # Detect start building
    pre_step_building = _detect_start_building(start_point.lat, start_point.lon)
    
    # Detect connector step (if distance to first edge is >10m)
    pre_step_connector = None
    if coords:
        from shapely.geometry import Point as ShapelyPoint
        first_coord = ShapelyPoint(coords[0])
        start_pt = ShapelyPoint(start_point.lon, start_point.lat)
        # Rough distance in degrees (good enough for short distances)
        dist_deg = first_coord.distance(start_pt)
        dist_m_approx = dist_deg * 111000  # approximate
        if dist_m_approx > 10 and edge_infos:
            first_edge = edge_infos[0]
            if first_edge.name != "path":
                pre_step_connector = f"Walk to {first_edge.name}"
            elif first_edge.facility_type:
                pre_step_connector = f"Walk to the {first_edge.facility_type}"

    # Merge edges into segments based on name and bearing continuity
    segments: List[EdgeInfo] = []
    if not edge_infos:
        return [], line, total_m, pre_step_building, pre_step_connector

    current_seg = EdgeInfo(
        name=edge_infos[0].name,
        facility_type=edge_infos[0].facility_type,
        bearing=edge_infos[0].bearing,
        length_m=edge_infos[0].length_m,
        data=edge_infos[0].data,
    )

    for i in range(1, len(edge_infos)):
        edge = edge_infos[i]
        delta = _bearing_delta(current_seg.bearing, edge.bearing)
        turn = _turn_phrase(delta)
        
        # Debug: print bearing changes
        if os.getenv("DEBUG_TURNS"):
            delta_str = f"{delta:.1f}°" if delta is not None else "N/A"
            bear_str = f"{current_seg.bearing:.1f}°→{edge.bearing:.1f}°" if current_seg.bearing and edge.bearing else "N/A"
            print(f"Edge {i}: {current_seg.name}→{edge.name}, bearing {bear_str}, Δ={delta_str}, turn={turn}")
        
        # Should we split?
        if edge.name != current_seg.name or turn is not None:
            segments.append(current_seg)
            current_seg = EdgeInfo(
                name=edge.name,
                facility_type=edge.facility_type,
                bearing=edge.bearing,
                length_m=edge.length_m,
                data=edge.data,
            )
        else:
            # Merge into current segment
            current_seg.length_m += edge.length_m
            # Keep original bearing for accurate cumulative turn detection
            # DO NOT update bearing - this was causing turns to be hidden!

    segments.append(current_seg)
    
    # Debug: show initial segments created
    if os.getenv("DEBUG_TURNS"):
        print(f"\n=== Initial segments: {len(segments)} ===")
        for i, seg in enumerate(segments):
            bear_str = f"{seg.bearing:.1f}°" if seg.bearing else "N/A"
            print(f"  Seg {i}: {seg.name}, bearing={bear_str}, length={seg.length_m:.1f}m")

    # Now merge short segments (< 20m) unless moderate turn (≥ 60°)
    merged_segments: List[EdgeInfo] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        # Look ahead to see if next segment is short
        if i + 1 < len(segments):
            next_seg = segments[i + 1]
            delta = _bearing_delta(seg.bearing, next_seg.bearing)
            ad = abs(delta) if delta is not None else 0
            # Merge if next segment is short and turn is not strong
            if next_seg.length_m < 20 and ad < 60:
                # Merge next into current
                seg.length_m += next_seg.length_m
                # Keep original bearing for cumulative turn detection
                i += 1  # skip next
                continue
        merged_segments.append(seg)
        i += 1

    segments = merged_segments

    # Debounce zig-zags: detect alternating slight turns within ~20m span
    # (simplified heuristic: look for consecutive segments with opposite slight turns)
    debounced: List[EdgeInfo] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        # Check if this and next form a zig-zag pattern
        if i + 1 < len(segments):
            next_seg = segments[i + 1]
            if seg.length_m + next_seg.length_m < 20:
                if i > 0:
                    prev_seg = debounced[-1] if debounced else None
                    if prev_seg:
                        delta1 = _bearing_delta(prev_seg.bearing, seg.bearing)
                        delta2 = _bearing_delta(seg.bearing, next_seg.bearing)
                        if delta1 and delta2:
                            # Opposite slight turns
                            if (abs(delta1) < 30 and abs(delta2) < 30 and
                                ((delta1 < 0 and delta2 > 0) or (delta1 > 0 and delta2 < 0))):
                                # Merge all three
                                prev_seg.length_m += seg.length_m + next_seg.length_m
                                # Keep original bearing for cumulative turn detection
                                i += 2
                                continue
        debounced.append(seg)
        i += 1

    segments = debounced

    # Convert segments into DirectionSteps with proper phrasing
    steps: List[DirectionStep] = []
    cumulative_m = 0.0

    for idx, seg in enumerate(segments):
        is_first = (idx == 0)
        is_last = (idx == len(segments) - 1)
        
        # Determine the instruction text
        if is_first:
            # First step: "Walk on <name>" with initial direction
            # Add cardinal direction for orientation
            cardinal = _cardinal_from_bearing(seg.bearing)
            direction_prefix = f" {cardinal.lower()}" if cardinal else ""
            
            if seg.name == "path":
                if seg.facility_type:
                    heading = f"Walk{direction_prefix} on the {seg.facility_type}"
                else:
                    heading = f"Walk{direction_prefix}"
            else:
                heading = f"Walk{direction_prefix} on {seg.name}"
        else:
            # Not first: check if we need a turn or continuation
            prev_seg = segments[idx - 1]
            delta = _bearing_delta(prev_seg.bearing, seg.bearing)
            turn = _turn_phrase(delta)
            
            if turn is None and seg.name == prev_seg.name:
                # Continue on same path
                if seg.name == "path":
                    heading = "Continue"
                else:
                    heading = f"Continue on {seg.name}"
            elif turn is None:
                # Name changed but no turn (e.g., transitioning between streets)
                if seg.name == "path":
                    heading = "Continue"
                else:
                    heading = f"Continue onto {seg.name}"
            else:
                # Turn detected
                if seg.name == "path":
                    # Avoid "onto path"
                    if seg.facility_type:
                        heading = f"Turn {turn} onto the {seg.facility_type}"
                    else:
                        heading = f"Turn {turn}"
                else:
                    heading = f"Turn {turn} onto {seg.name}"
        
        # Format with or without distance
        if hide_step_distances:
            text = heading
        else:
            text = f"{heading} for {_format_distance_m(seg.length_m, units)}"
        
        steps.append(
            DirectionStep(
                text=text,
                distance_m=seg.length_m,
                start_m=cumulative_m,
                end_m=cumulative_m + seg.length_m,
            )
        )
        cumulative_m += seg.length_m

    return steps, line, total_m, pre_step_building, pre_step_connector


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


def _clean_poi_name(name: str) -> str:
    """Strip parenthetical disambiguation like '(Newton)' from POI names."""
    # Remove trailing parenthetical like "(Newton)" or "(Building)"
    cleaned = re.sub(r'\s*\([^)]+\)\s*$', '', name)
    return cleaned.strip()


def _filter_poi_categories(
    df: pd.DataFrame,
    preferred_categories: Optional[str],
    blocked_categories: Optional[str],
) -> pd.DataFrame:
    """Filter POI dataframe by category preferences.
    
    Default: prefer campus/venue categories, drop generic natural/man_made/utilities.
    """
    if df.empty or "category" not in df.columns:
        return df
    
    # Parse CSV lists
    preferred = set()
    blocked = set()
    
    if preferred_categories:
        preferred = {c.strip().lower() for c in preferred_categories.split(",") if c.strip()}
    else:
        # Default preferred
        preferred = {
            "building", "library", "museum", "stadium", "theatre", "theater",
            "college", "university", "school", "chapel", "church", "monument",
            "memorial", "artwork", "statue", "fountain", "park", "garden",
        }
    
    if blocked_categories:
        blocked = {c.strip().lower() for c in blocked_categories.split(",") if c.strip()}
    else:
        # Default blocked
        blocked = {
            "natural", "man_made", "utility", "utilities", "broadcast",
            "antenna", "mast", "tower", "water", "power",
        }
    
    def should_include(cat: str) -> bool:
        cat_lower = str(cat).strip().lower()
        if not cat_lower:
            return True  # no category = include
        # Block if in blocked list
        for b in blocked:
            if b in cat_lower:
                return False
        # If we have preferred list, boost those
        # (but don't exclude others unless explicitly blocked)
        return True
    
    df = df[df["category"].apply(lambda c: should_include(str(c)))].copy()
    return df


def select_and_order_pois(
    df: pd.DataFrame,
    route_wgs84: LineString,
    max_pois: int,
    preferred_categories: Optional[str] = None,
    blocked_categories: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[PoiCallout]]:
    """Choose top POIs near the path and order them along the walk.

    We balance intrinsic score with perpendicular distance to keep mentions relevant.
    Now includes category filtering and name cleaning.
    """

    if df.empty or route_wgs84.is_empty:
        return df.head(0), []

    # Filter by categories
    df = _filter_poi_categories(df, preferred_categories, blocked_categories)
    if df.empty:
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
        
        # Clean the name
        raw_name = str(row.get("name") or "Unnamed")
        cleaned_name = _clean_poi_name(raw_name)
        
        callouts.append(
            PoiCallout(
                name=cleaned_name,
                description=description,
                along_m=float(row["along_m"]),
                side=side,
            )
        )

    return df, callouts


def weave_pois_into_steps(
    steps: List[DirectionStep],
    callouts: List[PoiCallout],
    max_callouts_per_step: int = 1,
    callout_style: str = "minimal",
) -> List[str]:
    """Produce final narrative lines: steps plus nearby POI callouts inline.

    We align POIs to the step whose distance range contains the POI position; if none, attach to the next step.
    Now respects max_callouts_per_step and callout_style.
    """

    lines: List[str] = []
    step_iter = iter(steps)
    current = next(step_iter, None)
    callout_idx = 0
    step_num = 1
    while current is not None:
        lines.append(f"{step_num}. {current.text}")
        # Emit POIs that fall within this step's span (up to max per step)
        step_callout_count = 0
        while callout_idx < len(callouts) and step_callout_count < max_callouts_per_step:
            c = callouts[callout_idx]
            if c.along_m <= current.end_m + 1e-6:
                side = f" on your {c.side}" if c.side else ""
                if callout_style == "descriptive":
                    # Include full description with category
                    lines.append(f"   - You'll pass{side} {c.name} ({c.description}).")
                else:
                    # Minimal: just name, drop category unless it's useful
                    lines.append(f"   - You'll pass{side} {c.name}.")
                callout_idx += 1
                step_callout_count += 1
            else:
                break
        current = next(step_iter, None)
        step_num += 1
    # Any remaining callouts go at the end; this is rare but keeps content.
    while callout_idx < len(callouts):
        c = callouts[callout_idx]
        side = f" on your {c.side}" if c.side else ""
        if callout_style == "descriptive":
            lines.append(f"   - Ahead{side}: {c.name} ({c.description}).")
        else:
            lines.append(f"   - Ahead{side}: {c.name}.")
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
    pre_step_building: Optional[str] = None,
    pre_step_connector: Optional[str] = None,
    units: str = "imperial",
) -> str:
    """Assemble the final plain-text tour document with pre-steps and arrival line."""

    header = [
        f"Walking tour from: {start.label} ({start.lat}, {start.lon})",
        f"Destination: {end.label} ({end.lat}, {end.lon})",
        f"Total distance: {_format_distance_m(total_m, units)}",
        f"Estimated time: {_estimate_walk_time_minutes(total_m)} min",
        "",
        "Directions:",
        "",
    ]
    
    body = []
    
    # Add pre-steps if present
    if pre_step_building:
        body.append(f"Walk out of {pre_step_building}.")
        body.append("")
    if pre_step_connector:
        body.append(f"{pre_step_connector}.")
        body.append("")
    
    # Add main directions
    for line in callout_lines:
        body.append(line)
    
    # Add arrival line
    body.append("")
    body.append(f"Arrive at {end.label}. This is your destination!")
    
    footer = ["", "Enjoy your walk!"]
    return "\n".join(header + body + footer)


# ---------------------------
# LLM post-processing
# ---------------------------


def _count_turns(text_lines: List[str]) -> int:
    """Count the number of turn instructions in direction text.
    
    Looks for turn-related words like 'left', 'right', 'turn', 'bear', 'sharp', etc.
    """
    turn_count = 0
    turn_keywords = [
        'left', 'right', 'turn', 'bear', 'sharp', 'u-turn',
        'slight left', 'slight right', 'bear left', 'bear right',
        'sharp left', 'sharp right'
    ]
    
    for line in text_lines:
        line_lower = line.lower()
        # Count each turn keyword occurrence
        for keyword in turn_keywords:
            if keyword in line_lower:
                turn_count += 1
                break  # Count only once per line
    
    return turn_count


def _validate_llm_turns(original_steps: List[DirectionStep], llm_lines: List[str]) -> bool:
    """Validate that LLM output preserves most turns from original steps.
    
    Returns True if LLM preserved at least 70% of turns, False otherwise.
    """
    # Count turns in original steps
    original_texts = [step.text for step in original_steps]
    original_turn_count = _count_turns(original_texts)
    
    # Count turns in LLM output
    llm_turn_count = _count_turns(llm_lines)
    
    # If there were no turns originally, validation passes
    if original_turn_count == 0:
        return True
    
    # Check if LLM preserved at least 70% of turns
    preservation_ratio = llm_turn_count / original_turn_count
    
    if preservation_ratio < 0.7:
        print(f"Warning: LLM output may have lost turns. "
              f"Original: {original_turn_count} turns, LLM: {llm_turn_count} turns "
              f"({preservation_ratio:.1%} preserved)")
        return False
    
    return True


def _llm_rewrite_directions(
    steps: List[DirectionStep],
    callouts: List[PoiCallout],
    start_label: str,
    end_label: str,
    args: argparse.Namespace,
) -> Optional[Tuple[List[str], str]]:
    """Use an LLM to rewrite directions into more natural language.
    
    Returns (rewritten_lines, arrival_text) or None on failure.
    Caches results to avoid redundant API calls.
    """
    
    # Build payload
    # Collect all words from steps for the allowed names list
    step_words = set()
    for s in steps:
        step_words.update(s.text.split())
    
    payload = {
        "steps": [
            {"id": i+1, "text": s.text, "distance_m": s.distance_m}
            for i, s in enumerate(steps)
        ],
        "pois": [
            {"id": i+1, "name": c.name, "side": c.side, "along_m": c.along_m, "description": c.description}
            for i, c in enumerate(callouts)
        ],
        "options": {
            "maxSteps": args.llm_max_steps,
            "includeDistances": args.llm_include_distances,
            "complexity": args.llm_complexity,
        },
        "labels": {
            "start": start_label,
            "end": end_label,
        },
        "allowed_names": [start_label, end_label] + [c.name for c in callouts] + list(step_words),
    }
    
    # Check cache
    payload_json = json.dumps(payload, sort_keys=True)
    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    cache_dir = os.path.join("cache", "llm")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{payload_hash}.json")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print("Using cached LLM response.")
            lines = [item["text"] for item in cached.get("steps", [])]
            arrival = cached.get("arrival", f"Arrive at {end_label}. This is your destination!")
            # Validate cached results too
            _validate_llm_turns(steps, lines)
            return lines, arrival
        except Exception as e:
            print(f"Failed to load cache: {e}")
    
    # Call LLM
    try:
        # Lazy import to allow running without openai installed
        try:
            from openai import OpenAI
        except ImportError:
            print("Warning: openai package not installed. Install with: pip install openai")
            return None
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not set. Skipping LLM rewriting.")
            return None
        
        client = OpenAI(api_key=api_key)
        
        # Select system prompt based on complexity level
        complexity = args.llm_complexity
        
        if complexity == "simple":
            system_prompt = (
                "You rewrite walking directions to be clear, natural, and human-friendly. "
                "Keep directions concise but conversational - avoid robotic phrasing. "
                "CRITICAL: Preserve all turns and directional changes from the original steps. "
                "Do not simplify actual turns into 'walk straight' or 'continue' - if the input has a turn, include it. "
                "Include initial compass direction (N/S/E/W) or relative direction for the first step. "
                "Only use names and places provided in the input. "
                "Do not fabricate new landmarks, street names, or buildings. "
                "Merge micro-steps into smooth, natural instructions only when they maintain the same direction. "
                "Keep the total number of steps small (ideally ≤8) but preserve all turns. "
                "Use warm, natural language like 'Head down X', 'Take a left onto Y', 'Continue along Z'. "
                "Vary your phrasing to sound more human and less mechanical. "
                "Make it feel like friendly directions from a local, not a GPS. "
                "Format: return JSON with 'steps' (array of {id, text}) and 'arrival' (string)."
            )
        elif complexity == "medium":
            system_prompt = (
                "You rewrite walking directions to be clear and helpful with moderate detail. "
                "CRITICAL: Preserve all turns and directional changes from the original steps. "
                "Do not simplify actual turns into 'walk straight' or 'continue' - if the input has a turn, include it. "
                "Include initial compass direction (N/S/E/W) or relative direction for the first step. "
                "Include helpful context and key landmarks to orient the walker. "
                "Only use names and places provided in the input. "
                "Do not fabricate new landmarks, street names, or buildings. "
                "Merge micro-steps into natural instructions only when they maintain the same direction. "
                "Keep around 8-12 steps with useful context, ensuring all turns are preserved. "
                "Mention notable POIs when they help with navigation. "
                "Use friendly, conversational language. "
                "Format: return JSON with 'steps' (array of {id, text}) and 'arrival' (string)."
            )
        else:  # complex
            system_prompt = (
                "You rewrite walking directions as a rich, detailed narrative tour. "
                "CRITICAL: Preserve all turns and directional changes from the original steps. "
                "Do not simplify actual turns into 'walk straight' or 'continue' - if the input has a turn, include it. "
                "Include initial compass direction (N/S/E/W) or relative direction for the first step. "
                "Create an engaging walking experience with vivid descriptions and context. "
                "Include surrounding landmarks, architectural details, and interesting facts about places passed. "
                "Only use names and places provided in the input - enhance their descriptions but don't invent new ones. "
                "Do not fabricate new landmarks, street names, or buildings. "
                "Weave POI callouts naturally into the narrative. "
                "Use descriptive, evocative language that helps the walker visualize the route. "
                "Can be longer (10-15 steps) to provide a fuller experience while preserving all directional changes. "
                "Make it feel like a guided tour, not just navigation instructions. "
                "Format: return JSON with 'steps' (array of {id, text}) and 'arrival' (string)."
            )
        
        
        user_prompt = (
            f"Rewrite these walking directions into natural language:\n\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            f"Return JSON with 'steps' and 'arrival' fields only."
        )
        
        response = client.chat.completions.create(
            model=args.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=args.llm_temperature,
            response_format={"type": "json_object"},
        )
        
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        # Save to cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
        lines = [item["text"] for item in result.get("steps", [])]
        arrival = result.get("arrival", f"Arrive at {end_label}. This is your destination!")
        
        # Validate that turns were preserved
        _validate_llm_turns(steps, lines)
        
        return lines, arrival
        
    except Exception as e:
        print(f"LLM rewriting failed: {e}")
        return None


# ---------------------------
# Main flow
# ---------------------------


def run(argv: Optional[Sequence[str]] = None) -> int:
    """Entrypoint for generating the tour.

    We separate execution into small, testable functions for readability and maintainability.
    """

    args = parse_args(argv)
    try:
        # Geocode with label overrides
        start = geocode_point(args.start, label_override=args.start_label)
        end = geocode_point(args.end, label_override=args.end_label)

        G = load_walk_graph(start, end, args.graph_distance_m)
        route = compute_route(G, start, end)
        steps, route_line, total_m, pre_step_building, pre_step_connector = build_directions(
            G, route, start, hide_step_distances=args.hide_step_distances, units=args.units
        )

        corridor = build_corridor_polygon(route_line, buffer_meters=int(args.buffer_meters))
        poi_df = fetch_scored_pois_in_corridor(
            corridor,
            min_score=float(args.min_poi_score),
            enable_google_places=bool(args.enable_google_places),
            google_api_key=args.google_api_key,
            google_radius_m=int(args.google_radius_meters),
        )
        _, callouts = select_and_order_pois(
            poi_df,
            route_line,
            max_pois=int(args.max_pois),
            preferred_categories=args.preferred_categories,
            blocked_categories=args.blocked_categories,
        )
        
        # Optional LLM post-processing
        if args.llm_enabled:
            llm_result = _llm_rewrite_directions(steps, callouts, start.label, end.label, args)
            if llm_result:
                narrative_lines, arrival_line = llm_result
                # Use LLM-generated text
                text = format_tour_text(
                    start, end, steps, narrative_lines, total_m,
                    pre_step_building=pre_step_building,
                    pre_step_connector=pre_step_connector,
                    units=args.units,
                )
                # Replace the auto-generated arrival with LLM's
                text = text.replace(
                    f"Arrive at {end.label}. This is your destination!",
                    arrival_line
                )
            else:
                # Fallback to rule-based
                narrative_lines = weave_pois_into_steps(
                    steps, callouts,
                    max_callouts_per_step=args.max_callouts_per_step,
                    callout_style=args.callout_style,
                )
                text = format_tour_text(
                    start, end, steps, narrative_lines, total_m,
                    pre_step_building=pre_step_building,
                    pre_step_connector=pre_step_connector,
                    units=args.units,
                )
        else:
            # Rule-based only
            narrative_lines = weave_pois_into_steps(
                steps, callouts,
                max_callouts_per_step=args.max_callouts_per_step,
                callout_style=args.callout_style,
            )
            text = format_tour_text(
                start, end, steps, narrative_lines, total_m,
                pre_step_building=pre_step_building,
                pre_step_connector=pre_step_connector,
                units=args.units,
            )

        out_path = os.path.abspath(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote walking tour to: {out_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
