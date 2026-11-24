from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from backend.app.core.config import settings, ROOT_DIR
from backend.app.db import init_db, get_db, Incident, Approval
from backend.app.schemas import (
    AnalyzeRequest,
    ApprovalRequest,
    FloodPredictionRequest,
    KnowledgeSearchRequest,
    TrafficPredictionRequest,
)
from backend.app.llm import get_llm_status
from backend.app.ml import get_models_status, predict_flood_risk, predict_traffic_congestion
from backend.app.traffic_data import get_traffic_context
from backend.app.workflow import run_workflow
from backend.app.weather import get_weather_context
from backend.app.rag import KB_DIR, get_rag_status, rebuild_knowledge_index, search_knowledge

init_db()
app=FastAPI(title=settings.app_name,version="0.2.0",description="Smart city multi-agent decision-support prototype; not for operational emergency use.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_origin_regex=settings.cors_origin_regex if settings.app_env.lower() == "development" else None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def demo_readings(scenario):
    values={
      "heavy_rain":{"rainfall_mm_hour":85,"traffic_density_pct":90,"water_level_m":1.2,"aqi":165,"drainage_capacity_pct":45,"source":"simulated_demo"},
      "traffic_peak":{"rainfall_mm_hour":5,"traffic_density_pct":92,"water_level_m":0.1,"aqi":95,"drainage_capacity_pct":75,"source":"simulated_demo"},
      "poor_air_quality":{"rainfall_mm_hour":0,"traffic_density_pct":55,"water_level_m":0.1,"aqi":190,"drainage_capacity_pct":75,"source":"simulated_demo"},
      "normal":{"rainfall_mm_hour":2,"traffic_density_pct":25,"water_level_m":0.1,"aqi":55,"drainage_capacity_pct":85,"source":"simulated_demo"},
      "custom":{"rainfall_mm_hour":0,"traffic_density_pct":0,"water_level_m":0,"aqi":0,"drainage_capacity_pct":60,"source":"user_supplied"}
    }
    return values[scenario]

@app.get("/")
def root(): return {"message":"CITYPULSE AI API is running","docs":"/docs"}

@app.get(settings.api_prefix+"/health")
def health(): return {"status":"ok","timestamp":datetime.now(timezone.utc).isoformat()}

@app.get(settings.api_prefix+"/demo/scenario")
def scenario(): return {"scenario":"heavy_rain","zone":"Zone A","readings":demo_readings("heavy_rain"),"note":"Synthetic demonstration values."}

@app.get(settings.api_prefix+"/models/status")
def models_status():
    return get_models_status()

@app.get(settings.api_prefix+"/llm/status")
def llm_status():
    return get_llm_status()

@app.get(settings.api_prefix+"/system/status")
def system_status():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": get_models_status(),
        "rag": get_rag_status(),
        "llm": get_llm_status(),
        "traffic_api": {
            "enabled": settings.traffic_api_enabled,
            "provider": settings.traffic_api_provider,
            "api_key_configured": bool(settings.traffic_api_key.strip()),
            "fallback_mode": "simulated_fallback",
        },
        "weather_api": {
            "enabled": settings.open_meteo_enabled,
            "provider": "Open-Meteo",
        },
    }

@app.post(settings.api_prefix+"/predict/traffic")
def predict_traffic(req: TrafficPredictionRequest):
    return predict_traffic_congestion(req.model_dump(), data_source=req.data_source, zone=req.zone)

@app.post(settings.api_prefix+"/predict/flood")
def predict_flood(req: FloodPredictionRequest):
    return predict_flood_risk(req.model_dump(), data_source=req.data_source, zone=req.zone)

@app.get(settings.api_prefix+"/traffic/live")
async def live_traffic(
    zone: str = "Zone A",
    use_live: bool = True,
    rainfall_mm: float = 0.0,
    special_event: int = 0,
    latitude: float | None = None,
    longitude: float | None = None,
):
    ctx = await get_traffic_context(
        zone=zone,
        use_live=use_live,
        latitude=latitude,
        longitude=longitude,
        rainfall_mm=rainfall_mm,
        special_event=special_event,
    )
    ml_pred = predict_traffic_congestion(
        ctx["ml_features"],
        data_source=ctx["data_origin"],
        zone=zone,
    )
    return {
        **ctx,
        "ml_prediction": ml_pred,
    }

@app.post(settings.api_prefix+"/analyze")
async def analyze(req: AnalyzeRequest, db: Session=Depends(get_db)):
    readings=dict(req.readings.model_dump() if req.readings else demo_readings(req.scenario))
    weather=None
    if req.use_live_weather:
        try: weather=await get_weather_context()
        except Exception as exc: weather={"error":str(exc),"message":"Weather connector failed; analysis continues with supplied readings."}
    traffic_context=None
    if req.use_live_traffic:
        traffic_context=await get_traffic_context(
            zone=req.zone,
            use_live=True,
            rainfall_mm=float(readings.get("rainfall_mm_hour", 0.0)),
            special_event=1 if req.scenario == "traffic_peak" else 0,
        )
        readings["traffic_density_pct"] = traffic_context["metrics"]["congestion_pct"]
        readings["traffic_source"] = traffic_context["data_origin"]
    traffic_features = req.traffic_features.model_dump() if req.traffic_features else None
    flood_features = req.flood_features.model_dump() if req.flood_features else None
    result=run_workflow({
        "scenario":req.scenario,
        "zone":req.zone,
        "readings":readings,
        "traffic_features":traffic_features,
        "flood_features":flood_features,
        "traffic_context":traffic_context,
        "weather_context":weather,
        "use_llm":req.use_llm,
    })
    incident_id=str(uuid4())
    summary=result["communication_finding"]["findings"]["report_summary"]
    db.add(Incident(id=incident_id,zone=req.zone,scenario=req.scenario,severity=result["risk_assessment"]["severity"],risk_score=result["risk_assessment"]["overall_score"],status="awaiting_review",summary=summary,result_json=result))
    db.commit()
    agent_keys = [
        "supervisor_finding",
        "traffic_finding",
        "flood_finding",
        "environment_finding",
        "infrastructure_finding",
        "risk_finding",
        "knowledge_finding",
        "response_finding",
        "communication_finding",
    ]
    llm_nodes = [k for k in ["supervisor_finding", "risk_finding", "response_finding", "communication_finding"] if result[k].get("reasoning_mode") == "llm"]
    return {
        "incident_id":incident_id,
        "zone":req.zone,
        "scenario":req.scenario,
        "readings":readings,
        "weather_context":weather,
        "traffic_context":traffic_context,
        "data_provenance":{
            "sensor_readings_source": readings.get("source", "simulated_demo"),
            "traffic_data_origin": traffic_context["data_origin"] if traffic_context else readings.get("source", "simulated_demo"),
            "traffic_is_simulated": traffic_context["is_simulated"] if traffic_context else True,
            "weather_is_live": bool(weather and weather.get("enabled") and "error" not in weather),
            "ml_training_data": "synthetic",
        },
        "llm_summary":{
            "llm_used": len(llm_nodes) > 0,
            "llm_nodes_count": len(llm_nodes),
            "reasoning_mode": "llm" if llm_nodes else "deterministic_fallback",
            "provider": settings.llm_provider,
            "model": settings.llm_model,
        },
        "risk_assessment":result["risk_assessment"],
        "agent_findings":[result[k] for k in agent_keys],
        "response_plan":result["response_plan"],
        "workflow_trace":result["workflow_trace"],
        "status":"awaiting_human_review",
        "disclaimer":"Prototype only. No real-world action was executed.",
    }

@app.get(settings.api_prefix+"/incidents")
def incidents(limit:int=20,db:Session=Depends(get_db)):
    rows=db.query(Incident).order_by(Incident.created_at.desc()).limit(min(max(limit,1),100)).all()
    return [{"id":x.id,"zone":x.zone,"scenario":x.scenario,"severity":x.severity,"risk_score":x.risk_score,"status":x.status,"summary":x.summary,"created_at":x.created_at.isoformat() if x.created_at else ""} for x in rows]

@app.post(settings.api_prefix+"/approvals/{incident_id}")
def approve(incident_id:str,req:ApprovalRequest,db:Session=Depends(get_db)):
    row=db.get(Incident,incident_id)
    if not row: raise HTTPException(404,"Incident not found")
    db.add(Approval(id=str(uuid4()),incident_id=incident_id,reviewer=req.reviewer,decision=req.decision,comment=req.comment))
    row.status=req.decision
    db.commit()
    return {"incident_id":incident_id,"decision":req.decision,"status":row.status}

@app.get(settings.api_prefix+"/knowledge/status")
def knowledge_status():
    return get_rag_status()

@app.post(settings.api_prefix+"/knowledge/search")
def knowledge_search(req:KnowledgeSearchRequest):
    results = search_knowledge(req.query, req.top_k)
    status = get_rag_status()
    return {
        "query": req.query,
        "rag_mode": status.get("active_mode", settings.rag_mode),
        "embedding_backend": status.get("embedding_backend", "unknown"),
        "indexed_chunks": status.get("indexed_chunks", 0),
        "results": results,
    }

@app.post(settings.api_prefix+"/knowledge/ingest")
async def knowledge_ingest(file:UploadFile=File(...)):
    suffix=Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf",".txt",".md"}: raise HTTPException(400,"Only PDF, TXT and MD files supported.")
    content=await file.read()
    if len(content)>10*1024*1024: raise HTTPException(413,"Maximum upload size is 10 MB.")
    raw_name = Path(file.filename or "document").name
    if suffix==".pdf":
        from pypdf import PdfReader
        import io
        try:
            reader=PdfReader(io.BytesIO(content))
            text="\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise HTTPException(400, f"Invalid or unreadable PDF file: {exc}") from exc
        target = KB_DIR / (Path(raw_name).stem + ".txt")
    else:
        text=content.decode("utf-8",errors="replace")
        target = KB_DIR / raw_name
    if not text.strip(): raise HTTPException(400,"No extractable text.")
    target.write_text(text,encoding="utf-8")
    index_status = rebuild_knowledge_index()
    return {
        "status": "ingested",
        "filename": target.name,
        "characters": len(text),
        "rag_mode": index_status.get("active_mode"),
        "indexed_chunks": index_status.get("indexed_chunks"),
    }
