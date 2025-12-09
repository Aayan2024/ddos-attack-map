from fastapi import FastAPI
import httpx
from typing import Optional
import json
import os
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pathlib import Path

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory cache for geolocation data
geo_cache = {}
CACHE_FILE = "geo_cache.json"

def load_cache():
    """Load cache from file on startup"""
    global geo_cache
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            geo_cache = json.load(f)
    print(f"Loaded {len(geo_cache)} cached IPs")

def save_cache():
    """Save cache to file"""
    with open(CACHE_FILE, 'w') as f:
        json.dump(geo_cache, f, indent=2)

# Load cache on startup
load_cache()

@app.get("/")
async def root():
    return {"message": "Live DDoS Attack Map API is running"}

@app.get("/geolocate/{ip_address}")
async def geolocate(ip_address: str):
    """
    Fetch geolocation data for a given IP address.
    Uses ipinfo.io API (free tier: 50k requests/month).
    """
    
    # Check cache first
    if ip_address in geo_cache:
        print(f"Cache hit for {ip_address}")
        return geo_cache[ip_address]
    
    try:
        # Fetch from ipinfo.io (free API)
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://ipinfo.io/{ip_address}/json")
            response.raise_for_status()
            data = response.json()
        
        # Parse location
        loc = data.get("loc", "0,0").split(",")
        latitude = float(loc[0]) if len(loc) > 0 else 0
        longitude = float(loc[1]) if len(loc) > 1 else 0
        
        # Structure response
        result = {
            "ip": ip_address,
            "latitude": latitude,
            "longitude": longitude,
            "country": data.get("country", "Unknown"),
            "city": data.get("city", "Unknown"),
            "region": data.get("region", "Unknown"),
            "org": data.get("org", "Unknown"),
            "cached": False
        }
        
        # Cache the result
        geo_cache[ip_address] = result
        save_cache()
        
        print(f"Fetched geolocation for {ip_address}: {result['city']}, {result['country']}")
        return result
    
    except httpx.HTTPError as e:
        return {
            "error": f"Failed to fetch geolocation for {ip_address}",
            "details": str(e),
            "ip": ip_address
        }

@app.get("/batch-geolocate")
async def batch_geolocate(ips: str):
    """
    Fetch geolocation for multiple IPs at once.
    
    Usage: /batch-geolocate?ips=8.8.8.8,1.1.1.1,208.67.222.222
    """
    ip_list = [ip.strip() for ip in ips.split(",")]
    results = []
    
    for ip in ip_list:
        result = await geolocate(ip)
        results.append(result)
    
    return {"count": len(results), "results": results}

@app.get("/cache-stats")
async def cache_stats():
    """Get cache statistics"""
    return {
        "cached_ips": len(geo_cache),
        "cache_file": CACHE_FILE,
        "sample_ips": list(geo_cache.keys())[:10]
    }

@app.delete("/cache-clear")
async def clear_cache():
    """Clear all cached data"""
    global geo_cache
    geo_cache = {}
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    return {"message": "Cache cleared"}

# AbuseIPDB Configuration
ABUSEIPDB_API_KEY = "f998ea3d158dde74cb1a8325dc4ca36557d3ce7952f074c69777d6840b5a3de9cf9c567cd328177a"
ABUSEIPDB_BASE_URL = "https://api.abuseipdb.com/api/v2"

@app.get("/fetch-attacks")
async def fetch_attacks(limit: int = 50):
    """
    Fetch most reported IPs from AbuseIPDB blacklist.
    These are IPs with high abuse confidence scores (potential DDoS sources).
    
    Returns: List of attack IPs with geolocation data
    """
    try:
        headers = {
            'Key': ABUSEIPDB_API_KEY,
            'Accept': 'application/json'
        }
        
        # Fetch blacklist (most reported IPs)
        params = {
            'confidenceMinimum': 70,  # Only high-confidence attackers
            'limit': limit
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ABUSEIPDB_BASE_URL}/blacklist",
                headers=headers,
                params=params,
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
        
        # Extract IPs and get geolocation for each
        attack_ips = []
        if 'data' in data:
            for item in data['data'][:limit]:
                ip = item.get('ipAddress')
                if ip:
                    # Get geolocation for this IP
                    geo_data = await geolocate(ip)
                    if geo_data and not geo_data.get('error'):
                        geo_data['abuseConfidence'] = item.get('abuseConfidenceScore', 0)
                        attack_ips.append(geo_data)
        
        return {
            "success": True,
            "count": len(attack_ips),
            "attacks": attack_ips
        }
    
    except httpx.HTTPError as e:
        return {
            "success": False,
            "error": f"Failed to fetch attacks from AbuseIPDB: {str(e)}",
            "attacks": []
        }
    
@app.get("/fetch-attacks-ml")
async def fetch_attacks_ml(limit: int = 20):
    """
    Fetch attacks from AbuseIPDB and add ML predictions for each IP.
    
    Returns: List of attack IPs with geolocation + ML predictions
    """
    try:
        # First fetch attacks normally
        attacks_data = await fetch_attacks(limit)
        
        if not attacks_data.get('success'):
            return attacks_data
        
        # Add ML predictions to each attack
        enhanced_attacks = []
        for attack in attacks_data['attacks']:
            ip = attack['ip']
            
            # Get ML prediction
            if ml_model and country_encoder:
                try:
                    country_code = attack.get('country', 'Unknown')
                    
                    # Encode country
                    try:
                        country_encoded = country_encoder.transform([country_code])[0]
                    except:
                        country_encoded = -1
                    
                    features = np.array([[
                        attack.get('latitude', 0),
                        attack.get('longitude', 0),
                        attack.get('abuseConfidence', 0),
                        country_encoded,
                        1 if attack.get('cached', False) else 0
                    ]])
                    
                    # Predict
                    prediction = ml_model.predict(features)[0]
                    probability = ml_model.predict_proba(features)[0]
                    
                    attack['ml_prediction'] = {
                        "is_ddos": bool(prediction),
                        "confidence": float(probability[1] * 100),
                        "risk_level": "High" if probability[1] >= 0.7 else "Medium" if probability[1] >= 0.4 else "Low"
                    }
                except Exception as e:
                    attack['ml_prediction'] = {
                        "error": str(e),
                        "is_ddos": None,
                        "confidence": 0,
                        "risk_level": "Unknown"
                    }
            else:
                attack['ml_prediction'] = {
                    "error": "ML model not loaded",
                    "is_ddos": None,
                    "confidence": 0,
                    "risk_level": "Unknown"
                }
            
            enhanced_attacks.append(attack)
        
        return {
            "success": True,
            "count": len(enhanced_attacks),
            "attacks": enhanced_attacks,
            "ml_enabled": ml_model is not None
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to fetch attacks with ML: {str(e)}",
            "attacks": []
        }

@app.get("/check-ip/{ip_address}")
async def check_ip(ip_address: str):
    """
    Check a specific IP against AbuseIPDB for abuse reports.
    
    Returns detailed abuse info including confidence score.
    """
    try:
        headers = {
            'Key': ABUSEIPDB_API_KEY,
            'Accept': 'application/json'
        }
        
        params = {
            'ipAddress': ip_address,
            'maxAgeInDays': 90
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ABUSEIPDB_BASE_URL}/check",
                headers=headers,
                params=params,
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
        
        # Combine with geolocation
        geo_data = await geolocate(ip_address)
        
        if 'data' in data:
            abuse_info = data['data']
            return {
                "ip": ip_address,
                "geolocation": geo_data,
                "abuseConfidenceScore": abuse_info.get('abuseConfidenceScore', 0),
                "totalReports": abuse_info.get('totalReports', 0),
                "isWhitelisted": abuse_info.get('isWhitelisted', False),
                "usageType": abuse_info.get('usageType', 'Unknown')
            }
        
        return {"error": "No data found"}
    
    except httpx.HTTPError as e:
        return {
            "error": f"Failed to check IP: {str(e)}"
        }
    
MODEL_PATH = Path("ddos_model.pkl")
ENCODER_PATH = Path("country_encoder.pkl")

ml_model = None
country_encoder = None

if MODEL_PATH.exists() and ENCODER_PATH.exists():
    ml_model = joblib.load(MODEL_PATH)
    country_encoder = joblib.load(ENCODER_PATH)
    print("✅ ML model loaded successfully")
else:
    print("⚠️ ML model not found. Run train_model.py first.")

@app.get("/predict-ddos/{ip_address}")
async def predict_ddos(ip_address: str):
    """
    Use ML model to predict DDoS likelihood for an IP address.
    
    Returns prediction confidence (0-100%)
    """
    if not ml_model or not country_encoder:
        return {
            "error": "ML model not loaded. Train model first.",
            "ip": ip_address
        }
    
    try:
        # Get geolocation data
        geo_data = await geolocate(ip_address)
        
        if geo_data.get('error'):
            return geo_data
        
        # Prepare features
        country_code = geo_data.get('country', 'Unknown')
        
        # Encode country
        try:
            country_encoded = country_encoder.transform([country_code])[0]
        except:
            country_encoded = -1  # Unknown country
        
        features = np.array([[
            geo_data.get('latitude', 0),
            geo_data.get('longitude', 0),
            geo_data.get('abuseConfidence', 0),
            country_encoded,
            1 if geo_data.get('cached', False) else 0
        ]])
        
        # Predict
        prediction = ml_model.predict(features)[0]
        probability = ml_model.predict_proba(features)[0]
        
        return {
            "ip": ip_address,
            "geolocation": geo_data,
            "ml_prediction": {
                "is_ddos": bool(prediction),
                "confidence": float(probability[1] * 100),  # Probability of being DDoS
                "risk_level": "High" if probability[1] >= 0.7 else "Medium" if probability[1] >= 0.4 else "Low"
            }
        }
    
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "ip": ip_address
        }
