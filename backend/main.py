import os
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml_utils import *

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODELS_ROOT = "../models"
models_cache = {}

CUSTOM_OBJECTS = {
    "BioLinearCKN1D": BioLinearCKN1D,
    "BioPolynomialCKN1D": BioPolynomialCKN1D,
    "BioNormalizedPolynomialCKN1D": BioNormalizedPolynomialCKN1D,
    "BioSphericalCKN1D": BioSphericalCKN1D,
    "BioGaussianCKN1D": BioGaussianCKN1D,
}


@app.on_event("startup")
def load_models():
    print("\n🔌 Loading BIO-CKN models")

    for ds in ["Ecoli", "ALKBH5", "RBP24", "DeepM6A"]:
        models_cache[ds] = {}
        path = os.path.join(MODELS_ROOT, ds)

        if not os.path.exists(path):
            print(f"⚠️ {ds} missing — using dummy models")
            for k in ["Linear", "Polynomial", "Spherical", "Gaussian"]:
                models_cache[ds][k] = create_dummy_model()
            continue

        for file in os.listdir(path):
            if not file.endswith(".keras"):
                continue

            name = file.lower()
            if "polynomial" in name:
                key = "Polynomial"
            elif "spherical" in name:
                key = "Spherical"
            elif "gaussian" in name:
                key = "Gaussian"
            else:
                key = "Linear"

            model = tf.keras.models.load_model(
                os.path.join(path, file),
                custom_objects=CUSTOM_OBJECTS
            )

            models_cache[ds][key] = model
            print(f"✅ Loaded {ds}/{file}  ({model.count_params():,} params)")

    print("🚀 Backend ready\n")


class PredictRequest(BaseModel):
    dataset: str
    model_type: str
    input_text: str


@app.get("/config")
def config():
    return {
        "datasets": list(models_cache.keys()),
        "models": ["Linear", "Polynomial", "Spherical", "Gaussian"]
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if req.dataset not in models_cache:
        raise HTTPException(404, "Dataset not found")

    model = models_cache[req.dataset].get(req.model_type)
    if model is None:
        raise HTTPException(404, "Model type not available")

    return extract_viz_data(model, req.input_text, req.dataset, req.model_type)
