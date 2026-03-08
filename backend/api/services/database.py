from pymongo import MongoClient
from datetime import datetime

client = MongoClient("mongodb://localhost:27017")

db = client["tradeeval"]

collection = db["results"]

def save_result(result):

    result["timestamp"] = datetime.utcnow()

    collection.insert_one(result)