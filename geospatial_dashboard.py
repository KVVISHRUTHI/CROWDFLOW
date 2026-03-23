import os
from typing import Any, Dict, List

import requests
from flask import Flask, jsonify, render_template, request

from processing.geospatial_intelligence import evaluate_zone, load_latest_ai_prediction, polygon_centroid

app = Flask(__name__, template_folder="templates")

PREDICTION_LOG_PATH = "prediction_output_log.csv"
HTTP_TIMEOUT_SEC = 20

GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
GOOGLE_PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"

NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1"


def _maps_provider() -> str:
    provider = os.getenv("MAPS_PROVIDER", "osm").strip().lower()
    if provider not in {"osm", "google"}:
        return "osm"
    return provider


def _headers() -> Dict[str, str]:
    return {
        "User-Agent": "CrowdSystemAI/1.0 (crowd geospatial intelligence)",
    }


@app.route("/")
def index():
    google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    return render_template(
        "geospatial_dashboard.html",
        google_maps_api_key=google_maps_api_key,
        maps_provider=_maps_provider(),
    )


@app.route("/api/ai/latest", methods=["GET"])
def api_ai_latest():
    latest = load_latest_ai_prediction(PREDICTION_LOG_PATH)
    if latest is None:
        return jsonify({
            "available": False,
            "message": "No prediction output file available yet. Run main.py first.",
        })

    return jsonify({
        "available": True,
        "ai": latest,
    })


def _get_google_maps_key() -> str:
    return os.getenv("GOOGLE_MAPS_API_KEY", "").strip()


def _call_google_api(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    key = _get_google_maps_key()
    if not key:
        return {"ok": False, "error": "GOOGLE_MAPS_API_KEY is not set."}

    enriched_params = dict(params)
    enriched_params["key"] = key

    try:
        response = requests.get(url, params=enriched_params, timeout=HTTP_TIMEOUT_SEC, headers=_headers())
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {"ok": False, "error": f"Google API request failed: {exc}"}

    status = payload.get("status", "UNKNOWN")
    if status not in {"OK", "ZERO_RESULTS"}:
        message = payload.get("error_message", "No additional error details provided.")
        return {"ok": False, "error": f"Google API returned status '{status}': {message}"}

    return {"ok": True, "payload": payload}


def _call_nominatim_reverse(lat: float, lng: float) -> Dict[str, Any]:
    params = {
        "format": "jsonv2",
        "lat": lat,
        "lon": lng,
        "addressdetails": 1,
    }
    try:
        response = requests.get(NOMINATIM_REVERSE_URL, params=params, timeout=HTTP_TIMEOUT_SEC, headers=_headers())
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {"ok": False, "error": f"Nominatim reverse geocode failed: {exc}"}

    if payload.get("error"):
        return {"ok": False, "error": payload.get("error")}

    return {"ok": True, "payload": payload}


def _call_nominatim_search(query: str, limit: int = 5) -> Dict[str, Any]:
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": max(1, min(limit, 25)),
        "addressdetails": 1,
        "dedupe": 0,
    }
    try:
        response = requests.get(NOMINATIM_SEARCH_URL, params=params, timeout=HTTP_TIMEOUT_SEC, headers=_headers())
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {"ok": False, "error": f"Nominatim search failed: {exc}"}

    return {"ok": True, "payload": payload}


def _extract_coords(payload: Dict[str, Any]) -> List[Dict[str, float]]:
    coords = payload.get("coordinates", [])
    out = []
    for point in coords:
        if "lat" not in point or "lng" not in point:
            continue
        out.append({"lat": float(point["lat"]), "lng": float(point["lng"])})
    return out


@app.route("/api/maps/health", methods=["GET"])
def api_maps_health():
    provider = _maps_provider()
    if provider == "google":
        return jsonify({
            "provider": "google",
            "google_maps_key_configured": bool(_get_google_maps_key()),
            "enabled_features": [
                "Maps JavaScript API",
                "Places API",
                "Geocoding API",
                "Directions API",
            ],
            "message": "Google provider enabled. Billing and API activation are required.",
        })

    return jsonify({
        "provider": "osm",
        "google_maps_key_configured": False,
        "enabled_features": [
            "OpenStreetMap tiles (Leaflet)",
            "Nominatim geocoding/reverse",
            "Overpass nearby places",
            "OSRM routing",
        ],
        "message": "Free map stack enabled. No billing required.",
    })


@app.route("/api/geospatial/evaluate", methods=["POST"])
def api_geospatial_evaluate():
    payload = request.get_json(force=True, silent=True) or {}
    coords = _extract_coords(payload)

    if len(coords) < 3:
        return jsonify({"error": "At least 3 coordinates are required."}), 400

    current_count = payload.get("current_count", 0)
    predicted_count = payload.get("predicted_count", 0)
    capacity_context = str(payload.get("capacity_context", "mixed_urban")).strip().lower()

    result = evaluate_zone(
        coords,
        int(current_count),
        int(predicted_count),
        capacity_context=capacity_context,
    )
    return jsonify(result)


@app.route("/api/geospatial/geocode", methods=["POST"])
def api_geocode():
    payload = request.get_json(force=True, silent=True) or {}
    query = str(payload.get("query", "")).strip()
    autocomplete_mode = bool(payload.get("autocomplete", False))
    default_limit = 12 if autocomplete_mode else 5
    limit = int(payload.get("limit", default_limit) or default_limit)
    limit = max(1, min(limit, 25))
    if not query:
        return jsonify({"error": "query is required."}), 400

    if _maps_provider() == "google":
        api_result = _call_google_api(GOOGLE_GEOCODE_URL, {"address": query})
        if not api_result["ok"]:
            return jsonify({"error": api_result["error"]}), 400

        results = api_result["payload"].get("results", [])
        if not results:
            return jsonify({"status": "ZERO_RESULTS", "results": []})

        mapped_results = []
        for item in results[:limit]:
            location = item.get("geometry", {}).get("location", {})
            mapped_results.append({
                "display_name": item.get("formatted_address", "Unknown"),
                "lat": location.get("lat"),
                "lng": location.get("lng"),
            })

        if not mapped_results:
            return jsonify({"status": "ZERO_RESULTS", "results": []})

        return jsonify({
            "status": "OK",
            "results": mapped_results,
        })

    api_result = _call_nominatim_search(query=query, limit=limit)
    if not api_result["ok"]:
        return jsonify({"error": api_result["error"]}), 400

    items = api_result["payload"]
    results = []
    for item in items:
        results.append({
            "display_name": item.get("display_name", "Unknown"),
            "lat": float(item.get("lat", 0.0)),
            "lng": float(item.get("lon", 0.0)),
        })

    return jsonify({
        "status": "OK" if results else "ZERO_RESULTS",
        "results": results,
    })


@app.route("/api/geospatial/reverse-geocode", methods=["POST"])
def api_reverse_geocode():
    payload = request.get_json(force=True, silent=True) or {}
    coords = _extract_coords(payload)

    if len(coords) < 3:
        return jsonify({"error": "At least 3 coordinates are required."}), 400

    centroid = polygon_centroid(coords)

    if _maps_provider() == "google":
        location = f"{centroid['lat']},{centroid['lng']}"
        api_result = _call_google_api(GOOGLE_GEOCODE_URL, {"latlng": location})
        if not api_result["ok"]:
            return jsonify({"error": api_result["error"]}), 400

        results = api_result["payload"].get("results", [])
        if not results:
            return jsonify({
                "centroid": centroid,
                "formatted_address": "No address found",
                "place_id": None,
            })

        top = results[0]
        return jsonify({
            "centroid": centroid,
            "formatted_address": top.get("formatted_address", "Unknown"),
            "place_id": top.get("place_id"),
            "types": top.get("types", []),
        })

    api_result = _call_nominatim_reverse(centroid["lat"], centroid["lng"])
    if not api_result["ok"]:
        return jsonify({"error": api_result["error"]}), 400

    item = api_result["payload"]
    return jsonify({
        "centroid": centroid,
        "formatted_address": item.get("display_name", "Unknown"),
        "place_id": item.get("place_id"),
        "types": [item.get("type", "unknown")],
    })


def _overpass_filter(place_type: str, keyword: str) -> List[str]:
    key = place_type.strip().lower()
    mapping = {
        "hospital": ['["amenity"="hospital"]'],
        "police": ['["amenity"="police"]'],
        "school": ['["amenity"="school"]'],
        "shopping_mall": ['["shop"="mall"]', '["building"="retail"]'],
        "transit_station": ['["public_transport"]', '["railway"="station"]', '["highway"="bus_stop"]'],
    }
    selector_parts = mapping.get(key, ['["amenity"]'])
    if keyword.strip():
        escaped = keyword.strip().replace('"', "")
        selector_parts = [f'{selector}["name"~"{escaped}",i]' for selector in selector_parts]

    selectors: List[str] = []
    for selector in selector_parts:
        selectors.append(f"node{selector}")
        selectors.append(f"way{selector}")
        selectors.append(f"relation{selector}")
    return selectors


def _build_overpass_query(selectors: List[str], radius: int, centroid: Dict[str, float]) -> str:
    body = "".join(
        f"{selector}(around:{radius},{centroid['lat']},{centroid['lng']});" for selector in selectors
    )
    return f"[out:json][timeout:25];({body});out center 60;"


def _extract_overpass_location(item: Dict[str, Any]) -> Dict[str, Any]:
    lat = item.get("lat")
    lng = item.get("lon")
    if lat is None or lng is None:
        center = item.get("center", {})
        lat = center.get("lat")
        lng = center.get("lon")
    return {"lat": lat, "lng": lng}


@app.route("/api/geospatial/nearby-places", methods=["POST"])
def api_nearby_places():
    payload = request.get_json(force=True, silent=True) or {}
    coords = _extract_coords(payload)

    if len(coords) < 3:
        return jsonify({"error": "At least 3 coordinates are required."}), 400

    centroid = polygon_centroid(coords)
    radius_m = int(payload.get("radius_m", 300))
    place_type = str(payload.get("place_type", "transit_station")).strip()
    keyword = str(payload.get("keyword", "")).strip()
    radius = max(50, min(radius_m, 20000))

    if _maps_provider() == "google":
        params: Dict[str, Any] = {
            "location": f"{centroid['lat']},{centroid['lng']}",
            "radius": radius,
            "type": place_type,
        }
        if keyword:
            params["keyword"] = keyword

        api_result = _call_google_api(GOOGLE_PLACES_NEARBY_URL, params)
        if not api_result["ok"]:
            return jsonify({"error": api_result["error"]}), 400

        raw_results = api_result["payload"].get("results", [])
        places = []
        for item in raw_results[:10]:
            location = item.get("geometry", {}).get("location", {})
            places.append({
                "name": item.get("name", "Unknown"),
                "vicinity": item.get("vicinity", "Unknown"),
                "rating": item.get("rating"),
                "user_ratings_total": item.get("user_ratings_total"),
                "types": item.get("types", []),
                "location": {
                    "lat": location.get("lat"),
                    "lng": location.get("lng"),
                },
                "place_id": item.get("place_id"),
            })
    else:
        selectors = _overpass_filter(place_type, keyword)
        candidate_radii = [radius, min(20000, max(radius * 2, 500)), min(20000, max(radius * 4, 1000))]
        payload_data = {"elements": []}
        last_error = None
        for candidate_radius in candidate_radii:
            query = _build_overpass_query(selectors, candidate_radius, centroid)
            try:
                response = requests.get(
                    OVERPASS_URL,
                    params={"data": query},
                    timeout=HTTP_TIMEOUT_SEC,
                    headers=_headers(),
                )
                response.raise_for_status()
                payload_data = response.json()
            except requests.RequestException as exc:
                last_error = exc
                continue

            if payload_data.get("elements"):
                radius = candidate_radius
                break

        if not payload_data.get("elements") and last_error is not None:
            return jsonify({"error": f"Overpass nearby search failed: {last_error}"}), 400

        elements = payload_data.get("elements", [])
        places = []
        for item in elements[:10]:
            tags = item.get("tags", {})
            location = _extract_overpass_location(item)
            if location["lat"] is None or location["lng"] is None:
                continue
            places.append({
                "name": tags.get("name", "Unnamed Place"),
                "vicinity": tags.get("addr:full", tags.get("addr:street", "Unknown vicinity")),
                "rating": None,
                "user_ratings_total": None,
                "types": [tags.get("amenity", tags.get("public_transport", "osm_place"))],
                "location": location,
                "place_id": str(item.get("id")),
            })

    return jsonify({
        "centroid": centroid,
        "radius_m": radius,
        "place_type": place_type,
        "keyword": keyword,
        "count": len(places),
        "places": places,
    })


def _extract_lat_lng(point: Dict[str, Any], field_name: str) -> Dict[str, float]:
    if "lat" not in point or "lng" not in point:
        raise ValueError(f"{field_name} must contain lat and lng fields.")
    return {
        "lat": float(point["lat"]),
        "lng": float(point["lng"]),
    }


@app.route("/api/geospatial/directions", methods=["POST"])
def api_directions():
    payload = request.get_json(force=True, silent=True) or {}
    origin = payload.get("origin")

    if not isinstance(origin, dict):
        return jsonify({"error": "origin object with lat/lng is required."}), 400

    try:
        origin_lat_lng = _extract_lat_lng(origin, "origin")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    destination = payload.get("destination")
    coords = _extract_coords(payload)
    if isinstance(destination, dict):
        try:
            destination_lat_lng = _extract_lat_lng(destination, "destination")
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    elif len(coords) >= 3:
        destination_lat_lng = polygon_centroid(coords)
    else:
        return jsonify({"error": "Provide destination or coordinates for centroid destination."}), 400

    travel_mode = str(payload.get("mode", "walking")).lower()
    if travel_mode not in {"walking", "driving", "bicycling", "transit"}:
        return jsonify({"error": "Invalid mode. Use walking/driving/bicycling/transit."}), 400

    if _maps_provider() == "google":
        params = {
            "origin": f"{origin_lat_lng['lat']},{origin_lat_lng['lng']}",
            "destination": f"{destination_lat_lng['lat']},{destination_lat_lng['lng']}",
            "mode": travel_mode,
        }
        api_result = _call_google_api(GOOGLE_DIRECTIONS_URL, params)
        if not api_result["ok"]:
            return jsonify({"error": api_result["error"]}), 400

        routes = api_result["payload"].get("routes", [])
        if not routes:
            return jsonify({
                "origin": origin_lat_lng,
                "destination": destination_lat_lng,
                "mode": travel_mode,
                "distance_text": "N/A",
                "duration_text": "N/A",
                "status": "ZERO_RESULTS",
                "route_geometry": [],
            })

        leg = routes[0].get("legs", [{}])[0]
        return jsonify({
            "origin": origin_lat_lng,
            "destination": destination_lat_lng,
            "mode": travel_mode,
            "distance_text": leg.get("distance", {}).get("text", "N/A"),
            "duration_text": leg.get("duration", {}).get("text", "N/A"),
            "start_address": leg.get("start_address", "Unknown"),
            "end_address": leg.get("end_address", "Unknown"),
            "status": "OK",
            "route_geometry": [],
        })

    profile_map = {
        "walking": "foot",
        "driving": "driving",
        "bicycling": "bike",
        "transit": "driving",
    }
    profile = profile_map[travel_mode]

    coordinates = (
        f"{origin_lat_lng['lng']},{origin_lat_lng['lat']};"
        f"{destination_lat_lng['lng']},{destination_lat_lng['lat']}"
    )
    url = f"{OSRM_ROUTE_URL}/{profile}/{coordinates}"
    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "false",
    }

    try:
        response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SEC, headers=_headers())
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return jsonify({"error": f"OSRM routing failed: {exc}"}), 400

    if data.get("code") != "Ok" or not data.get("routes"):
        return jsonify({
            "origin": origin_lat_lng,
            "destination": destination_lat_lng,
            "mode": travel_mode,
            "distance_text": "N/A",
            "duration_text": "N/A",
            "status": "ZERO_RESULTS",
            "route_geometry": [],
        })

    route = data["routes"][0]
    distance_m = float(route.get("distance", 0.0))
    duration_s = float(route.get("duration", 0.0))

    geometry = route.get("geometry", {}).get("coordinates", [])
    route_geometry = [{"lat": c[1], "lng": c[0]} for c in geometry if len(c) >= 2]

    return jsonify({
        "origin": origin_lat_lng,
        "destination": destination_lat_lng,
        "mode": travel_mode,
        "distance_text": f"{distance_m / 1000.0:.2f} km",
        "duration_text": f"{duration_s / 60.0:.1f} min",
        "start_address": f"{origin_lat_lng['lat']:.6f}, {origin_lat_lng['lng']:.6f}",
        "end_address": f"{destination_lat_lng['lat']:.6f}, {destination_lat_lng['lng']:.6f}",
        "status": "OK",
        "route_geometry": route_geometry,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
