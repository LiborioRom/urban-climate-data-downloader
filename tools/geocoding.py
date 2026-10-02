from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import requests
from pydantic import BaseModel, ConfigDict, Field


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


class GeocodingCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: str
    latitude: float
    longitude: float
    osm_type: str | None = None
    osm_id: int | None = None
    category: str | None = None
    place_type: str | None = None
    importance: float = 0.0
    bounding_box: tuple[float, float, float, float] | None = None
    geojson: dict[str, Any] | None = None


def _cache_path(query: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"{digest}.json"


def geocode_place(
    query: str,
    cache_dir: Path,
    *,
    requester: Callable[..., Any] = requests.get,
) -> list[GeocodingCandidate]:
    """Consulta Nominatim con caché; nunca permite que el LLM invente coordenadas."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = _cache_path(query, cache_dir)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8"))
    else:
        response = requester(
            NOMINATIM_URL,
            params={
                "q": query,
                "format": "jsonv2",
                "polygon_geojson": 1,
                "addressdetails": 1,
                "countrycodes": "es",
                "limit": 5,
            },
            headers={
                "User-Agent": os.getenv(
                    "NOMINATIM_USER_AGENT",
                    "lst-prediction-tfm/1.0 (local academic application)",
                )
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    candidates: list[GeocodingCandidate] = []
    for item in payload:
        bbox = item.get("boundingbox")
        candidates.append(GeocodingCandidate(
            display_name=item["display_name"],
            latitude=float(item["lat"]),
            longitude=float(item["lon"]),
            osm_type=item.get("osm_type"),
            osm_id=int(item["osm_id"]) if item.get("osm_id") is not None else None,
            category=item.get("category"),
            place_type=item.get("type"),
            importance=float(item.get("importance") or 0),
            bounding_box=(float(bbox[2]), float(bbox[0]), float(bbox[3]), float(bbox[1])) if bbox else None,
            geojson=item.get("geojson"),
        ))
    return candidates


def choose_candidate(candidates: list[GeocodingCandidate], query: str) -> GeocodingCandidate:
    if not candidates:
        raise LookupError(f"No se encontró ningún lugar para: {query}")
    query_words = {word for word in query.lower().replace(",", " ").split() if len(word) > 2}

    def score(candidate: GeocodingCandidate) -> tuple[float, float]:
        name = candidate.display_name.lower()
        word_score = sum(word in name for word in query_words)
        area_bonus = 2 if candidate.geojson and candidate.geojson.get("type") in {"Polygon", "MultiPolygon"} else 0
        locality_bonus = 1 if candidate.place_type in {"neighbourhood", "suburb", "quarter", "residential"} else 0
        return word_score + area_bonus + locality_bonus, candidate.importance

    return max(candidates, key=score)

