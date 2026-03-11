from google.cloud import storage
import joblib

def download_model(symbol):
    client = storage.Client()
    bucket = client.bucket("trading-models")
    blob = bucket.blob(f"{symbol}/model_latest.pkl")
    blob.download_to_filename(f"ml/models/{symbol}.pkl")

def load_model(symbol):
    return joblib.load(f"ml/models/{symbol}.pkl")