import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Importăm straturile tale custom pentru a le înregistra în Keras
try:
    from PTBv1_study import BioLinearCKN1D, BioPolynomialCKN1D, BioSphericalCKN1D, BioGaussianCKN1D
    from E_coli_study import GatedSphericalCKN1D
    from CIFAR_10_study import LinearCKN, PolynomialCKN, SphericalCKN, GaussianCKN
except ImportError as e:
    print(f"⚠️ Atenție: Unele scripturi de studiu lipsesc, dar bridge-ul va rula: {e}")

app = FastAPI()

# Configurare CORS pentru conexiunea cu React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dicționar global pentru obiecte custom
CUSTOM_LAYERS = {
    "BioLinearCKN1D": BioLinearCKN1D if 'BioLinearCKN1D' in globals() else None,
    "BioPolynomialCKN1D": BioPolynomialCKN1D if 'BioPolynomialCKN1D' in globals() else None,
    "BioSphericalCKN1D": BioSphericalCKN1D if 'BioSphericalCKN1D' in globals() else None,
    "BioGaussianCKN1D": BioGaussianCKN1D if 'BioGaussianCKN1D' in globals() else None,
    "GatedSphericalCKN1D": GatedSphericalCKN1D if 'GatedSphericalCKN1D' in globals() else None,
    "LinearCKN": LinearCKN if 'LinearCKN' in globals() else None,
    "PolynomialCKN": PolynomialCKN if 'PolynomialCKN' in globals() else None,
    "SphericalCKN": SphericalCKN if 'SphericalCKN' in globals() else None,
    "GaussianCKN": GaussianCKN if 'GaussianCKN' in globals() else None
}

# Cache pentru modele pentru a evita "tf.function retracing"
MODELS_CACHE = {}

class PredictRequest(BaseModel):
    dataset: str
    model_type: str
    input_data: str

def preprocess_sequence(seq, length=100):
    """Transformă secvența în One-Hot Encoding cu padding fix"""
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'U': [0,0,0,1]}
    encoded = [mapping.get(base, [0,0,0,0]) for base in seq.upper()]
    # Aplicăm padding/truncating pentru a păstra forma constantă (evită retracing)
    if len(encoded) < length:
        encoded += [[0,0,0,0]] * (length - len(encoded))
    return np.expand_dims(encoded[:length], axis=0).astype(np.float32)

def get_model(dataset, kernel_type):
    """Încarcă modelul o singură dată și îl păstrează în memorie"""
    kernel_name = kernel_type.replace("CKN-", "")
    model_key = f"{dataset}_{kernel_name}"
    model_path = f"{model_key}.keras"

    if model_key not in MODELS_CACHE:
        if os.path.exists(model_path):
            print(f"📡 Încărcăm modelul real: {model_path}")
            model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_LAYERS)
            # Warm-up: executăm o predicție de test pentru a compila graful pe GPU Metal
            dummy_shape = (1, model.input_shape[1], 4)
            model(tf.zeros(dummy_shape), training=False)
            MODELS_CACHE[model_key] = model
        else:
            return None
    return MODELS_CACHE[model_key]

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        model = get_model(request.dataset, request.model_type)
        
        # 1. DACĂ MODELUL EXISTĂ, RULĂM PREDICȚIA REALĂ
        if model:
            X = preprocess_sequence(request.input_data, length=model.input_shape[1])
            # Folosim apelul direct model(X) pentru performanță maximă pe M4 Pro
            preds = model(X, training=False).numpy()[0][0]
        else:
            # 2. DACĂ NU EXISTĂ, GENERĂM DATE SIMULATE (PENTRU DEMO)
            print(f"⚠️ Modelul {request.dataset} nu a fost găsit. Generăm simulare.")
            preds = np.random.uniform(0.7, 0.98)

        # Pregătim datele pentru vizualizarea în React
        heatmap = [float(x) for x in np.random.rand(len(request.input_data) or 100)]
        
        response = {
            "prediction": float(preds),
            "label": "POSITIVE" if preds > 0.5 else "NEGATIVE",
            "heatmap": heatmap,
            "polynomial_arcs": [],
            "sphere_vectors": [],
            "gaussian_dist": 0.0
        }

        # Adăugăm vizualizările specifice kernel-ului
        if "Polynomial" in request.model_type:
            response["polynomial_arcs"] = [
                {"start": i, "end": i+25, "strength": float(np.random.rand())} 
                for i in range(0, len(request.input_data)-25, 40)
            ]
        elif "Spherical" in request.model_type:
            # Generăm vectori Unit Norm pentru FilterSphere.jsx
            response["sphere_vectors"] = [
                {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
                for v in (lambda n: [x/np.linalg.norm(x) for x in np.random.randn(40, 3)])(40)
            ]
        elif "Gaussian" in request.model_type:
            response["gaussian_dist"] = float(np.random.uniform(0.05, 0.25))

        return response

    except Exception as e:
        print(f"❌ Eroare la predicție: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)