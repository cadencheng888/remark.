"""
Best-effort current location via IP geolocation — no OS permissions, no native
deps. Makes the agent location-aware: Claude resolves "near me"/"around here",
and the Browserbase web agent knows where the wearer is for "find X nearby",
directions, food delivery, local availability, etc.

Cached for an hour; degrades to None silently if offline. Override for demos:
    LOCATION_OVERRIDE="Mountain View, California, United States"
    LOCATION_LATLNG="37.3861,-122.0839"
"""
import json
import os
import time
import urllib.request

_TTL = 3600  # seconds
_cache = {"ts": 0.0, "data": None}

_OVERRIDE = os.environ.get("LOCATION_OVERRIDE", "").strip()
_OVERRIDE_LATLNG = os.environ.get("LOCATION_LATLNG", "").strip()


def _label(city, region, country) -> str | None:
    return ", ".join(p for p in (city, region, country) if p) or None


def _fetch() -> dict | None:
    try:
        with urllib.request.urlopen("http://ip-api.com/json/", timeout=4) as r:
            d = json.loads(r.read().decode())
        if d.get("status") != "success":
            return None
        city, region, country = d.get("city"), d.get("regionName"), d.get("country")
        return {
            "city": city, "region": region, "country": country,
            "lat": d.get("lat"), "lng": d.get("lon"),
            "label": _label(city, region, country),
        }
    except Exception:
        return None


def get_location() -> dict | None:
    """Return {city, region, country, lat, lng, label} or None. Cached."""
    if _OVERRIDE:
        lat = lng = None
        if "," in _OVERRIDE_LATLNG:
            try:
                lat, lng = (float(x.strip()) for x in _OVERRIDE_LATLNG.split(",", 1))
            except ValueError:
                pass
        return {"city": _OVERRIDE, "region": None, "country": None,
                "lat": lat, "lng": lng, "label": _OVERRIDE}

    now = time.monotonic()
    if _cache["data"] and now - _cache["ts"] < _TTL:
        return _cache["data"]
    data = _fetch()
    if data:
        _cache["data"] = data
        _cache["ts"] = now
    return data


def location_label() -> str | None:
    loc = get_location()
    return loc["label"] if loc else None
