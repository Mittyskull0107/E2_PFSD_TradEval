import os
from pymongo import MongoClient
from datetime import datetime


# ── lazy connection — doesn't crash Django if Mongo is down ───────
_client     = None
_db         = None
_collection = None


def _get_collection():
    """Connect to MongoDB lazily on first use."""
    global _client, _db, _collection

    if _collection is not None:
        return _collection

    mongo_uri = os.environ.get("MONGODB_URI", "mongodb://127.0.0.1:27017/")

    try:
        _client     = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        _client.server_info()   # force connection — raises if unreachable
        _db         = _client["tradeeval_db"]
        _collection = _db["results"]
        print(f"[database] Connected to MongoDB at {mongo_uri}")
    except Exception as e:
        print(f"[database] WARNING: Could not connect to MongoDB: {e}")
        _collection = None

    return _collection


def save_result(data: dict) -> bool:
    """
    Save a result document to MongoDB.

    Args:
        data: dict to save — type, symbol, result etc.

    Returns:
        True if saved successfully, False otherwise
    """
    collection = _get_collection()

    if collection is None:
        print("[database] WARNING: MongoDB unavailable — result not saved")
        return False

    try:
        data["timestamp"] = datetime.utcnow()
        collection.insert_one(data)
        return True
    except Exception as e:
        print(f"[database] ERROR saving result: {e}")
        return False


def get_results(result_type: str = None, limit: int = 50) -> list:
    """
    Retrieve saved results from MongoDB.

    Args:
        result_type : filter by type e.g. 'backtest', 'event_analysis', 'risk_prediction'
        limit       : max number of results to return

    Returns:
        list of result dicts (with _id removed for JSON safety)
    """
    collection = _get_collection()

    if collection is None:
        return []

    try:
        query  = {"type": result_type} if result_type else {}
        cursor = collection.find(query).sort("timestamp", -1).limit(limit)

        results = []
        for doc in cursor:
            doc.pop("_id", None)   # remove MongoDB ObjectId (not JSON serializable)
            results.append(doc)

        return results

    except Exception as e:
        print(f"[database] ERROR fetching results: {e}")
        return []
