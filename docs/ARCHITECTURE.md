# Architecture and Implementation Status

## Nine Agent Roles (LangGraph Multi-Agent Workflow)
1. **City Supervisor (`city_supervisor_agent`)** — Triages incoming events, identifies priority domains, and flags data provenance caveats (LLM-enhanced with deterministic fallback).
2. **Traffic Intelligence (`traffic_intelligence_agent`)** — Runs the trained Traffic Congestion Random Forest Regressor (`ml/traffic/models/traffic_model.joblib`) using real-time TomTom/HERE traffic telemetry, simulated fallback telemetry, or sensor readings.
3. **Weather & Flood (`weather_flood_agent`)** — Evaluates hydrological indicators and runs the trained Flood Risk ML Classifier (`ml/flood/models/flood_model.joblib`).
4. **Environmental (`environmental_agent`)** — Evaluates AQI readings and exposure bands.
5. **Infrastructure (`infrastructure_agent`)** — Reviews storm-drainage capacity and maintenance flags.
6. **Risk Assessment (`risk_assessment_agent`)** — Combines rule-based scores, Traffic RF ML, and Flood ML scores with LLM cross-domain cascading risk reasoning (with deterministic fallback).
7. **Knowledge / RAG (`knowledge_rag_agent`)** — Performs FAISS embedding-based semantic retrieval (`IndexFlatIP` on L2-normalized vectors, persisted under `data/vector_store/`, with optional ChromaDB and keyword modes) over SOP documents in `knowledge_base/`.
8. **Response Planning (`response_planning_agent`)** — Synthesizes prioritized, SOP-cited recommendations via LLM (or deterministic baseline) while strictly enforcing `requires_human_approval: true` and `execution_status: "recommendation_only"`.
9. **Communication (`communication_reporting_agent`)** — Produces executive operator briefings and `draft_only` citizen advisory drafts via LLM (or deterministic fallback).

## Data Provenance & Safety Boundaries
- Synthetic ML training datasets (`ml/traffic/datasets/traffic_data.csv`, `ml/flood/datasets/flood_data.csv`) are explicitly labeled across API payloads, model status endpoints, and the Next.js UI.
- Real-time traffic API (`TRAFFIC_API_ENABLED=true`) and Open-Meteo weather (`OPEN_METEO_ENABLED=true`) are clearly distinguished from `simulated_fallback` and `simulated_demo` readings.
- API keys (`TRAFFIC_API_KEY`, `LLM_API_KEY`) are loaded strictly from environment variables via Pydantic Settings and never hardcoded.
- Do not claim validated real-world accuracy or autonomous emergency dispatch until models are trained, calibrated, and validated on jurisdiction-specific real historical records.
