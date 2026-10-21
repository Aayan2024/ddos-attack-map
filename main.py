from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import asyncio
import ipaddress
import json
import logging
import math
import os

import httpx
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from backend.history import initialize_database, record_collection
from backend.reporting import router as reporting_router


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CACHE_FILE = BASE_DIR / "geo_cache.json"
MODEL_PATH = BASE_DIR / "ddos_model.pkl"
ENCODER_PATH = BASE_DIR / "country_encoder.pkl"

ABUSEIPDB_API_KEY = os.getenv("ABUSEIPDB_API_KEY", "").strip()
ABUSEIPDB_BASE_URL = "https://api.abuseipdb.com/api/v2"

# Cache clearing is disabled unless explicitly enabled locally.
ENABLE_CACHE_CLEAR = (
    os.getenv("ENABLE_CACHE_CLEAR", "false").lower() == "true"
)

DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:5500,"
    "http://localhost:5500"
)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        DEFAULT_CORS_ORIGINS,
    ).split(",")
    if origin.strip()
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ddos_api")

geo_cache = {}
ml_model = None
country_encoder = None


# ---------------------------------------------------------
# Input and normalization helpers
# ---------------------------------------------------------

def validate_ip(value: str) -> str:
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid IPv4 or IPv6 address",
        )


def normalize_coordinate(value, minimum, maximum):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    if not minimum <= number <= maximum:
        return None

    return number


def normalize_score(value):
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None

    return score if 0 <= score <= 100 else None


def location_record(ip_address: str, data: dict, cached=False):
    """
    Return only geolocation fields.

    Old reputation/prediction fields in geo_cache.json are
    deliberately excluded.
    """
    latitude = normalize_coordinate(
        data.get("latitude"), -90, 90
    )
    longitude = normalize_coordinate(
        data.get("longitude"), -180, 180
    )

    if latitude is None or longitude is None:
        latitude = longitude = None

    return {
        "ip": ip_address,
        "latitude": latitude,
        "longitude": longitude,
        "country": data.get("country") or "Unknown",
        "city": data.get("city") or "Unknown",
        "region": data.get("region") or "Unknown",
        "org": data.get("org") or "Unknown",
        "timezone": data.get("timezone"),
        "location_available": latitude is not None,
        "cached": cached,
    }


def unavailable_location(ip_address: str):
    return location_record(ip_address, {})


# ---------------------------------------------------------
# Cache management
# ---------------------------------------------------------

def load_cache():
    global geo_cache

    geo_cache = {}

    if not CACHE_FILE.exists():
        logger.info("No existing geolocation cache")
        return

    try:
        with CACHE_FILE.open("r", encoding="utf-8") as file:
            stored = json.load(file)

        if not isinstance(stored, dict):
            raise ValueError("Cache root must be a JSON object")

        for ip, value in stored.items():
            if not isinstance(value, dict):
                continue

            try:
                normalized_ip = str(ipaddress.ip_address(ip))
            except ValueError:
                continue

            geo_cache[normalized_ip] = location_record(
                normalized_ip,
                value,
            )

        logger.info("Loaded %s cached IPs", len(geo_cache))

    except (OSError, ValueError, TypeError):
        logger.exception(
            "Could not load geolocation cache; starting empty"
        )


def save_cache():
    temporary_path = CACHE_FILE.with_suffix(".json.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                geo_cache,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        os.replace(temporary_path, CACHE_FILE)

    except (OSError, ValueError, TypeError):
        logger.exception("Could not persist geolocation cache")


# ---------------------------------------------------------
# Experimental ML
# ---------------------------------------------------------

def load_ml_artifacts():
    global ml_model, country_encoder

    ml_model = None
    country_encoder = None

    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        logger.warning(
            "ML artifacts unavailable; non-ML endpoints remain usable"
        )
        return

    try:
        # Only load trusted local artifacts.
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)

        ml_model = model
        country_encoder = encoder

        logger.info("Experimental ML artifacts loaded")

    except Exception:
        logger.exception(
            "ML artifact loading failed; ML endpoints are unavailable"
        )


def prediction_unavailable(message: str):
    return {
        "is_ddos": None,
        "confidence": None,
        "risk_level": "Unknown",
        "experimental": True,
        "error": message,
    }


def predict_record(record: dict):
    """
    Compatibility wrapper for the existing threshold-derived model.

    The output is experimental, not validated DDoS detection.
    """
    if ml_model is None or country_encoder is None:
        return prediction_unavailable("ML model not loaded")

    latitude = record.get("latitude")
    longitude = record.get("longitude")
    abuse_score = normalize_score(record.get("abuseConfidence"))

    if latitude is None or longitude is None:
        return prediction_unavailable(
            "Prediction requires available coordinates"
        )

    if abuse_score is None:
        return prediction_unavailable(
            "Prediction requires a reputation score"
        )

    try:
        try:
            country_encoded = int(
                country_encoder.transform(
                    [record.get("country", "Unknown")]
                )[0]
            )
        except ValueError:
            country_encoded = -1

        # Match the exact feature order in train_model.py.
        # Legacy training data used cached=False throughout.
        # Pin this legacy feature to 0 so cache hits do not
        # change risk predictions.
        features = np.array([[
            latitude,
            longitude,
            abuse_score,
            country_encoded,
            0,
        ]], dtype=float)

        classes = list(ml_model.classes_)

        if len(classes) != 2 or set(classes) != {0, 1}:
            return prediction_unavailable(
                "Expected a binary model with labels 0 and 1"
            )

        positive_index = classes.index(1)
        probabilities = ml_model.predict_proba(features)[0]
        positive_probability = float(
            probabilities[positive_index]
        )
        predicted_label = ml_model.predict(features)[0]

        risk_level = (
            "High" if positive_probability >= 0.7
            else "Medium" if positive_probability >= 0.4
            else "Low"
        )

        return {
            # Retained for compatibility with the original frontend.
            "is_ddos": bool(predicted_label == 1),
            "confidence": positive_probability * 100,
            "risk_level": risk_level,
            "experimental": True,
            "interpretation": (
                "Legacy model score for a reputation-threshold label; "
                "not independently validated DDoS likelihood"
            ),
        }

    except Exception:
        logger.exception("Experimental prediction failed")
        return prediction_unavailable("Prediction failed")


# ---------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(initialize_database)
    load_cache()
    load_ml_artifacts()

    app.state.geo_lock = asyncio.Lock()
    app.state.geo_semaphore = asyncio.Semaphore(4)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=5.0),
        limits=httpx.Limits(
            max_connections=10,
            max_keepalive_connections=5,
        ),
        follow_redirects=False,
    ) as client:
        app.state.http_client = client
        yield


app = FastAPI(
    title="Cyber Intelligence Observatory API",
    version="0.2.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "frontend"),
    name="static",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(reporting_router)


# ---------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------

async def abuseipdb_request(endpoint: str, params: dict):
    if not ABUSEIPDB_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="ABUSEIPDB_API_KEY is not configured",
        )

    try:
        response = await app.state.http_client.get(
            f"{ABUSEIPDB_BASE_URL}/{endpoint}",
            headers={
                "Key": ABUSEIPDB_API_KEY,
                "Accept": "application/json",
            },
            params=params,
            timeout=30.0,
        )
        response.raise_for_status()

    except httpx.HTTPStatusError as exc:
        provider_status = exc.response.status_code

        if provider_status == 429:
            raise HTTPException(
                status_code=503,
                detail="AbuseIPDB request quota or rate limit reached",
            )

        if provider_status in (401, 403):
            raise HTTPException(
                status_code=502,
                detail=(
                    "AbuseIPDB rejected access; check the configured "
                    "key and endpoint entitlement"
                ),
            )

        raise HTTPException(
            status_code=502,
            detail=f"AbuseIPDB returned HTTP {provider_status}",
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to AbuseIPDB",
        )

    try:
        payload = response.json()
    except ValueError:
        raise HTTPException(
            status_code=502,
            detail="AbuseIPDB returned invalid JSON",
        )

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502,
            detail="Unexpected AbuseIPDB response format",
        )

    return payload


async def lookup_geolocation(ip_address: str):
    if ip_address in geo_cache:
        return location_record(
            ip_address,
            geo_cache[ip_address],
            cached=True,
        )

    async with app.state.geo_semaphore:
        # Another request may have populated the cache while waiting.
        if ip_address in geo_cache:
            return location_record(
                ip_address,
                geo_cache[ip_address],
                cached=True,
            )

        try:
            response = await app.state.http_client.get(
                f"https://ipinfo.io/{ip_address}/json"
            )
            response.raise_for_status()
            data = response.json()

        except httpx.HTTPError:
            raise HTTPException(
                status_code=502,
                detail="IPinfo geolocation request failed",
            )

        except ValueError:
            raise HTTPException(
                status_code=502,
                detail="IPinfo returned invalid JSON",
            )

        if not isinstance(data, dict):
            raise HTTPException(
                status_code=502,
                detail="Unexpected IPinfo response format",
            )

        if data.get("error"):
            raise HTTPException(
                status_code=502,
                detail="IPinfo could not provide geolocation",
            )

        latitude = longitude = None
        raw_location = data.get("loc")

        if isinstance(raw_location, str):
            parts = raw_location.split(",")

            if len(parts) == 2:
                latitude = normalize_coordinate(parts[0], -90, 90)
                longitude = normalize_coordinate(parts[1], -180, 180)

        result = location_record(
            ip_address,
            {
                **data,
                "latitude": latitude,
                "longitude": longitude,
            },
        )

        async with app.state.geo_lock:
            geo_cache[ip_address] = dict(result)
            save_cache()

        return dict(result)


async def fetch_indicator_collection(limit: int):
    payload = await abuseipdb_request(
        "blacklist",
        {
            "confidenceMinimum": 70,
            "limit": limit,
        },
    )

    source_items = payload.get("data")

    if not isinstance(source_items, list):
        raise HTTPException(
            status_code=502,
            detail="Unexpected AbuseIPDB blacklist format",
        )

    async def enrich(item):
        if not isinstance(item, dict):
            return None

        try:
            ip_address = str(
                ipaddress.ip_address(item.get("ipAddress", ""))
            )
        except (ValueError, TypeError):
            return None

        try:
            location = await lookup_geolocation(ip_address)
        except HTTPException:
            # Keep the intelligence observation even if location fails.
            location = unavailable_location(ip_address)
            location["geolocation_error"] = (
                "Geolocation enrichment unavailable"
            )

        return {
            **location,
            "abuseConfidence": normalize_score(
                item.get("abuseConfidenceScore")
            ),
        }

    results = await asyncio.gather(
        *(enrich(item) for item in source_items[:limit])
    )

    # Deduplicate within this collection, not across all history.
    unique = {
        item["ip"]: item
        for item in results
        if item is not None
    }
    indicators = list(unique.values())

    collection = await run_in_threadpool(
        record_collection,
        indicators,
    )

    return {
        "success": True,
        "count": len(indicators),
        # Original field retained for existing globe compatibility.
        "attacks": indicators,
        "collection": collection,
        "data_kind": "reported_ip_reputation",
    }


# ---------------------------------------------------------
# Basic and frontend routes
# ---------------------------------------------------------

@app.get("/globe", include_in_schema=False)
async def globe_page():
    path = BASE_DIR / "globe.html"

    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="globe.html was not found in the project root",
        )

    return FileResponse(path)
    
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/globe", status_code=307)

@app.get("/health")
async def health():
    return {
        "message": "Cyber intelligence API is running",
        "globe": "/globe",
        "docs": "/docs",
        "history": "/api/observations",
        "report": "/api/reports/weekly.xlsx",
        "ml_enabled": ml_model is not None,
    }

# ---------------------------------------------------------
# Geolocation routes
# ---------------------------------------------------------

@app.get("/geolocate/{ip_address}")
async def geolocate(ip_address: str):
    """Return approximate IP geolocation, when available."""
    return await lookup_geolocation(validate_ip(ip_address))


@app.get("/batch-geolocate")
async def batch_geolocate(
    ips: str = Query(..., min_length=1, max_length=5000),
):
    supplied = [value.strip() for value in ips.split(",")]

    if len(supplied) > 50:
        raise HTTPException(
            status_code=422,
            detail="At most 50 IPs are allowed per batch",
        )

    normalized = list(dict.fromkeys(
        validate_ip(value) for value in supplied
    ))

    async def safe_lookup(ip_address):
        try:
            return await lookup_geolocation(ip_address)
        except HTTPException as exc:
            return {
                "ip": ip_address,
                "error": exc.detail,
            }

    results = await asyncio.gather(
        *(safe_lookup(ip) for ip in normalized)
    )

    return {
        "count": len(results),
        "results": results,
    }


@app.get("/cache-stats")
async def cache_stats():
    return {
        "cached_ips": len(geo_cache),
        "cache_file": CACHE_FILE.name,
        "sample_ips": list(geo_cache)[:10],
    }


@app.delete("/cache-clear")
async def clear_cache():
    if not ENABLE_CACHE_CLEAR:
        raise HTTPException(
            status_code=403,
            detail="Cache clearing is disabled",
        )

    async with app.state.geo_lock:
        geo_cache.clear()
        save_cache()

    return {"message": "Geolocation cache cleared"}


# ---------------------------------------------------------
# Intelligence routes
# ---------------------------------------------------------

@app.get("/fetch-attacks")
async def fetch_attacks(
    limit: int = Query(20, ge=1, le=100),
):
    """Collect reputation indicators and persist observations."""
    return await fetch_indicator_collection(limit)


@app.get("/fetch-attacks-ml")
async def fetch_attacks_ml(
    limit: int = Query(20, ge=1, le=100),
):
    """Collect indicators and attach experimental legacy ML scores."""
    result = await fetch_indicator_collection(limit)

    for indicator in result["attacks"]:
        indicator["ml_prediction"] = await run_in_threadpool(
            predict_record,
            indicator,
        )

    result["ml_enabled"] = (
        ml_model is not None and country_encoder is not None
    )
    result["ml_warning"] = (
        "Experimental reputation-threshold model, "
        "not validated DDoS detection"
    )

    return result


@app.get("/check-ip/{ip_address}")
async def check_ip(ip_address: str):
    ip_address = validate_ip(ip_address)

    payload = await abuseipdb_request(
        "check",
        {
            "ipAddress": ip_address,
            "maxAgeInDays": 90,
        },
    )

    abuse_info = payload.get("data")

    if not isinstance(abuse_info, dict):
        raise HTTPException(
            status_code=502,
            detail="Unexpected AbuseIPDB check response",
        )

    try:
        location = await lookup_geolocation(ip_address)
    except HTTPException:
        location = unavailable_location(ip_address)
        location["geolocation_error"] = (
            "Geolocation enrichment unavailable"
        )

    return {
        "ip": ip_address,
        "geolocation": location,
        "abuseConfidenceScore": normalize_score(
            abuse_info.get("abuseConfidenceScore")
        ),
        "totalReports": abuse_info.get("totalReports"),
        "isWhitelisted": abuse_info.get("isWhitelisted"),
        "usageType": abuse_info.get("usageType") or "Unknown",
        "lastReportedAt": abuse_info.get("lastReportedAt"),
        "data_kind": "reported_ip_reputation",
    }


@app.get("/predict-ddos/{ip_address}")
async def predict_ddos(ip_address: str):
    ip_address = validate_ip(ip_address)

    if ml_model is None or country_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Experimental ML model is unavailable",
        )

    # Explicitly retrieve reputation instead of relying on cache state.
    intelligence = await check_ip(ip_address)

    record = {
        **intelligence["geolocation"],
        "abuseConfidence": intelligence["abuseConfidenceScore"],
    }

    prediction = await run_in_threadpool(
        predict_record,
        record,
    )

    return {
        "ip": ip_address,
        "geolocation": intelligence["geolocation"],
        "abuseConfidenceScore": intelligence["abuseConfidenceScore"],
        "ml_prediction": prediction,
        "warning": (
            "This legacy model learned a reputation-score threshold. "
            "Its confidence is not a validated probability of DDoS."
        ),
    }
