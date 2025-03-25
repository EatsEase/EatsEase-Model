from decouple import config
from motor.motor_asyncio import AsyncIOMotorClient

# MongoDB URI
MONGO_URL = config("MONGO_URL", cast=str, default="mongodb://localhost:27017")

# Async client
client = AsyncIOMotorClient(
    MONGO_URL,
    tls=True,
    tlsAllowInvalidCertificates=True  # ⚠️ only for dev/self-signed certs
)

# Get your database and collections
db = client["EatsEase"]
user_profile_collection = db["UserProfile"]
menu_collection = db["Menu"]
