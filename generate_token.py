import os
import pyotp
from dotenv import load_dotenv
from growwapi import GrowwAPI

load_dotenv()

API_KEY = os.getenv("GROWW_API_KEY")
API_SECRET = os.getenv("GROWW_SECRET")
TOTP_SECRET = os.getenv("GROWW_TOTP_SECRET")

if not API_KEY or not API_SECRET:
    raise Exception("Missing GROWW_API_KEY or GROWW_SECRET")

print("🔐 Generating Groww access token...")

totp_code = None
if TOTP_SECRET:
    totp_code = pyotp.TOTP(TOTP_SECRET).now()

try:

    token = GrowwAPI.get_access_token(
        api_key=API_KEY,
        secret=API_SECRET,
        totp=totp_code
    )

    with open("access_token.txt", "w") as f:
        f.write(token)

    print("✅ Token generated and saved")

except Exception as e:
    print("❌ Token generation failed:", e)
