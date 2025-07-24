from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import io
from modules.tools import predict_risk
from modules.data_processing import DataProcessing
from fastapi.responses import StreamingResponse

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI(
    title="Crédit pour Tous - API de Prédiction",
    description="API REST pour la prédiction des défauts de paiement",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClientData(BaseModel):
    dti: float = Field(..., ge=0, le=100)
    annual_inc: float = Field(..., ge=1000)
    loan_amnt: float = Field(..., ge=1000)
    int_rate: float = Field(..., ge=1, le=30)
    revol_util: float = Field(..., ge=0, le=100)

class PredictionResponse(BaseModel):
    risk_score: float
    risk_level: str
    recommendation: str
    confidence: float

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    analysis_type: str

class PredictSimpleRequest(BaseModel):
    dti: float
    annual_inc: float
    loan_amnt: float
    int_rate: float
    revol_util: float

class PredictSimpleResponse(BaseModel):
    score: float
    decision: str
    threshold: float
    base_value: float
    shap: list

@app.get("/", tags=["Health"])
async def root():
    logging.info("Endpoint / appelé.")
    return {
        "message": "Crédit pour Tous - API de Prédiction des Défauts",
        "version": "1.0.0",
        "status": "active",
        "endpoints": ["/docs", "/predict", "/chat", "/health"]
    }

@app.get("/health")
async def health():
    logging.info("Endpoint /health appelé.")
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_default(client_data: ClientData):
    try:
        data_dict = client_data.dict()
        logging.info(f"Requête de prédiction reçue : {data_dict}")
        risk_score, prediction = predict_risk(data_dict)
        if risk_score < 0.4:
            risk_level = "Faible"
            recommendation = "Acceptation recommandée"
        elif risk_score < 0.7:
            risk_level = "Modéré"
            recommendation = "Surveillance requise"
        else:
            risk_level = "Élevé"
            recommendation = "Conditions renforcées"
        confidence = min(0.95, abs(risk_score - 0.5) * 2)
        logging.info(f"Prédiction : score={risk_score}, niveau={risk_level}")
        return PredictionResponse(
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            recommendation=recommendation,
            confidence=round(confidence, 3)
        )
    except Exception as e:
        logging.error(f"Erreur de prédiction : {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erreur de prédiction: {str(e)}")

@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat_with_bot(chat_request: ChatRequest):
    try:
        logging.info(f"Message chatbot reçu : {chat_request.message}")
        response = "Fonctionnalité chatbot à compléter."
        analysis_type = "general"
        logging.info(f"Réponse chatbot : {response}")
        return ChatResponse(
            response=response,
            analysis_type=analysis_type
        )
    except Exception as e:
        logging.error(f"Erreur chatbot : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur chatbot: {str(e)}")

@app.post("/upload-dataset", tags=["Dataset"])
async def upload_dataset(file: UploadFile = File(...)):
    try:
        logging.info(f"Upload dataset : {file.filename}")
        if not file.filename.endswith('.csv'):
            logging.warning("Format de fichier non supporté.")
            raise HTTPException(status_code=400, detail="Format de fichier non supporté. Utilisez CSV.")
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        processed = DataProcessing(df).full_processing()
        temp_path = f"DATA/temp_{file.filename}"
        processed.to_csv(temp_path, index=False)
        logging.info(f"Dataset {file.filename} uploadé et nettoyé.")
        return {
            "filename": file.filename,
            "size": len(contents),
            "status": "uploaded",
            "message": "Dataset uploadé et nettoyé avec succès.",
            "columns": list(processed.columns)
        }
    except Exception as e:
        logging.error(f"Erreur upload : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur upload: {str(e)}")

@app.get("/model-info", tags=["Model"])
async def get_model_info():
    logging.info("Endpoint /model-info appelé.")
    return {
        "model_name": "LightGBM",
        "model_version": "1.0",
        "features_count": 15,
        "threshold": 0.40,
        "performance": {
            "accuracy": "66,7%",
            "precision": "27,7%",
            "recall": "81,2%",
            "f1_score": "41,3%"
        },
        "last_trained": "2024-07-24"
    }

@app.post("/predict-simple", response_model=PredictSimpleResponse, tags=["Simulation"])
async def predict_simple(request: PredictSimpleRequest):
    try:
        input_data = request.dict()
        logging.info(f"Simulation simple reçue : {input_data}")
        score, prediction = predict_risk(input_data)
        if score < 0.4:
            decision = "Acceptation recommandée"
        elif score < 0.7:
            decision = "Surveillance requise"
        else:
            decision = "Conditions renforcées"
        base_value = 0.0
        shap = []
        logging.info(f"Simulation : score={score}, décision={decision}")
        return PredictSimpleResponse(
            score=round(score, 4),
            decision=decision,
            threshold=0.4,
            base_value=base_value,
            shap=shap
        )
    except Exception as e:
        logging.error(f"Erreur de simulation : {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erreur de simulation: {str(e)}")

@app.post("/predict-batch", tags=["Batch"])
async def predict_batch(file: UploadFile = File(...)):
    try:
        logging.info(f"Batch reçu : {file.filename}")
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df_clean = DataProcessing(df).full_processing()
        # Prédire pour chaque client
        results = []
        for idx, row in df_clean.iterrows():
            input_data = row.to_dict()
            score, _ = predict_risk(input_data)
            if score < 0.4:
                decision = "Acceptation recommandée"
            elif score < 0.7:
                decision = "Surveillance requise"
            else:
                decision = "Conditions renforcées"
            results.append({
                "index": idx,
                "score": round(score, 4),
                "decision": decision
            })
        # Statistiques globales
        n_total = len(results)
        n_accept = sum(r["decision"] == "Acceptation recommandée" for r in results)
        n_surv = sum(r["decision"] == "Surveillance requise" for r in results)
        n_cond = sum(r["decision"] == "Conditions renforcées" for r in results)
        stats = {
            "total": n_total,
            "acceptation": n_accept,
            "surveillance": n_surv,
            "conditions_renforcees": n_cond
        }
        logging.info(f"Batch stats : {stats}")
        return {"stats": stats, "results": results}
    except Exception as e:
        logging.error(f"Erreur batch : {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erreur batch: {str(e)}")

@app.get("/dashboard-data", tags=["Dashboard"])
async def dashboard_data(id: str):
    try:
        file_path = f"DATA/{id}.csv"
        logging.info(f"Dashboard data demandé pour : {file_path}")
        if not os.path.exists(file_path):
            logging.warning("Job ID non trouvé.")
            raise HTTPException(status_code=404, detail="Job ID non trouvé")
        df = pd.read_csv(file_path)
        stats = {
            "total": len(df),
            "acceptation": int((df["decision"] == "Acceptation recommandée").sum()),
            "surveillance": int((df["decision"] == "Surveillance requise").sum()),
            "conditions_renforcees": int((df["decision"] == "Conditions renforcées").sum())
        }
        logging.info(f"Dashboard stats : {stats}")
        return {"id": id, "stats": stats}
    except Exception as e:
        logging.error(f"Erreur dashboard : {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erreur dashboard: {str(e)}")

@app.get("/download/{id}.csv", tags=["Dashboard"])
async def download_csv(id: str):
    file_path = f"DATA/{id}.csv"
    logging.info(f"Téléchargement demandé pour : {file_path}")
    if not os.path.exists(file_path):
        logging.warning("ID non trouvé pour téléchargement.")
        raise HTTPException(status_code=404, detail="ID non trouvé")
    with open(file_path, "rb") as f:
        csv_bytes = f.read()
    logging.info(f"Fichier {file_path} envoyé au client.")
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={id}.csv"}
    )

if __name__ == "__main__":
    logging.info("Lancement du serveur FastAPI avec Uvicorn.")
    import uvicorn
    uvicorn.run("Applications.api_app:app", host="127.0.0.1", port=8000, reload=True)

