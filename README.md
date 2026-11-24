# CITYPULSE AI
**Autonomous Smart City Intelligence & Operations Platform**

Full-stack smart-city decision-support prototype combining **FastAPI**, **LangGraph 9-agent orchestration with optional LLM reasoning**, **trained Traffic & Flood ML models**, **FAISS embedding-based RAG**, **real-time Traffic & Weather connectors with simulated fallback**, **SQLite/PostgreSQL persistence**, and a **Next.js mission-control dashboard**.

## Important Accuracy & Data Provenance Boundary
- **Synthetic ML Models**: Both the Traffic Congestion Random Forest Regressor (`ml/traffic/models/traffic_model.joblib`) and the Flood Risk ML Classifier (`ml/flood/models/flood_model.joblib`) are trained on **synthetic demonstration datasets**. Their evaluation metrics reflect synthetic holdout performance only and must **not** be presented as validated real-world accuracy.
- **Simulated vs. Live Data**: Every API response and dashboard banner explicitly tags data provenance (`real_time_api`, `simulated_fallback`, `simulated_demo`, `user_supplied`).
- **Human-in-the-Loop Safety**: All generated response plans remain `recommendation_only` with `requires_human_approval: true`, and communication outputs are `draft_only`. No autonomous dispatch or real-world emergency action is executed.

---

## Implemented Capabilities
1. **Traffic Congestion ML (Random Forest Regressor)**:
   - Reuses trained `ml/traffic/models/traffic_model.joblib` (7 features: `hour`, `day_of_week`, `vehicle_count`, `average_speed`, `road_capacity`, `rainfall_mm`, `special_event`).
   - Integrated into `POST /api/v1/predict/traffic`, LangGraph `traffic_agent`, and the Next.js ML Predictors dashboard.
2. **Flood Risk ML Model (Trained, Evaluated & Integrated)**:
   - Reproducible pipeline in `ml/flood/` (`generate_dataset.py`, `train_model.py`, `predict.py`).
   - Compares `LogisticRegression`, `RandomForestClassifier`, and `GradientBoostingClassifier` using 5-fold stratified cross-validation and held-out test metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier score, confusion matrix).
   - Saves best artifact to `ml/flood/models/flood_model.joblib` and full evaluation metadata to `ml/flood/models/flood_metrics.json`.
   - Integrated into `POST /api/v1/predict/flood`, LangGraph `flood_agent`, and the Next.js dashboard.
3. **Real-Time Traffic Data API with Simulated Fallback**:
   - `backend/app/traffic_data.py` supports **TomTom Traffic Flow API v4** and **HERE Traffic Flow API v7** via environment variables (`TRAFFIC_API_ENABLED`, `TRAFFIC_API_PROVIDER`, `TRAFFIC_API_KEY`).
   - Automatically falls back to deterministic time-of-day simulated telemetry (`data_origin: "simulated_fallback"`, `is_simulated: true`) when disabled, unconfigured, or unreachable.
   - Exposed via `GET /api/v1/traffic/live` and optionally wired into `POST /api/v1/analyze` (`use_live_traffic: true`).
4. **Embedding-Based RAG with FAISS (plus Chroma & Keyword Modes)**:
   - Upgraded `backend/app/rag.py` to chunk documents with markdown heading awareness and index L2-normalized dense embeddings in a persisted **FAISS (`IndexFlatIP`)** vector store (`data/vector_store/`).
   - Supports `sentence-transformers` when installed/cached locally, with an automatic offline-capable local dense LSA embedding backend, optional `RAG_MODE=chroma`, and legacy `RAG_MODE=keyword`.
   - Automatically re-indexes on `POST /api/v1/knowledge/ingest` (PDF, TXT, MD).
5. **LLM-Enhanced LangGraph Multi-Agent Workflow**:
   - `backend/app/llm.py` provides a provider-agnostic structured JSON LLM adapter supporting **OpenAI**, **Google Gemini**, **Ollama (local)**, and **OpenAI-compatible APIs** with strict Pydantic output validation.
   - Enhances the **City Supervisor**, **Risk Assessment**, **Response Planning**, and **Communication** agents inside the existing 9-agent LangGraph workflow, with guaranteed deterministic fallback when disabled or unavailable.
6. **Full-Stack Dashboard, Tests & Deployment**:
   - Next.js dashboard (`frontend/app/page.tsx`) with 4 operational views: Multi-Agent Operations (with custom sensors, live API/LLM toggles, data provenance bar, and human approval workflow), ML Predictors & Live Traffic, FAISS Knowledge RAG (search + upload), and Incident History & Approvals.
   - Comprehensive `pytest` suite (`backend/tests/`), `backend/Dockerfile`, `frontend/Dockerfile`, and full-stack `docker-compose.yml`.

---

## Quick Start — Windows PowerShell
Requirements: Python 3.11/3.12, Node.js 20+, npm.

1. Open a terminal inside `CITYPULSE-AI`:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   *(If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.)*
2. Install backend dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r backend/requirements.txt
   ```
3. Configure environment variables (never commit `.env` or real API keys):
   ```powershell
   Copy-Item .env.example .env
   ```
4. *(Optional)* Regenerate / retrain Flood ML model or test standalone ML predictors:
   ```powershell
   python ml/traffic/predict.py
   python ml/flood/generate_dataset.py
   python ml/flood/train_model.py
   python ml/flood/predict.py
   ```
5. Start FastAPI backend:
   ```powershell
   uvicorn backend.app.main:app --reload
   ```
   API docs: `http://127.0.0.1:8000/docs`
6. In a second terminal, start the Next.js dashboard:
   ```powershell
   cd frontend
   Copy-Item .env.example .env.local
   npm install
   npm run dev
   ```
   Dashboard: `http://localhost:3000`

---

## Environment Configuration (`.env`)
All secrets are loaded exclusively from environment variables via `backend/app/core/config.py`. Never hardcode API keys.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./data/citypulse.db` | SQLite default; supports PostgreSQL/PostGIS |
| `CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001` | Allowed frontend origins (any local dev port also allowed when `APP_ENV=development`) |
| `OPEN_METEO_ENABLED` | `false` | Enable live Open-Meteo precipitation context |
| `TRAFFIC_API_ENABLED` | `false` | Enable real-time traffic API connector |
| `TRAFFIC_API_PROVIDER` | `tomtom` | `tomtom`, `here`, or `simulated` |
| `TRAFFIC_API_KEY` | *(empty)* | TomTom or HERE API key (falls back to simulated if empty) |
| `RAG_MODE` | `faiss` | `faiss` (default), `chroma`, or `keyword` |
| `RAG_PERSIST_DIRECTORY` | `./data/vector_store` | Persisted FAISS/Chroma index directory |
| `LLM_ENABLED` | `false` | Enable LLM reasoning in LangGraph agents |
| `LLM_PROVIDER` | `openai` | `openai`, `gemini`, `ollama`, `openai_compatible`, or `disabled` |
| `LLM_MODEL` | `gpt-4o-mini` | Model identifier for the chosen provider |
| `LLM_API_KEY` | *(empty)* | Provider API key (not needed for local `ollama`) |
| `LLM_BASE_URL` | *(empty)* | Optional custom endpoint (e.g. Groq, OpenRouter, Ollama) |

---

## API Reference
- `GET /api/v1/health` — Liveness probe
- `GET /api/v1/system/status` — Aggregated status of ML models, FAISS RAG, LLM, and traffic/weather APIs
- `GET /api/v1/models/status` — Loaded Traffic & Flood ML model status and evaluation metrics
- `GET /api/v1/llm/status` — Active LLM provider readiness and fallback configuration
- `GET /api/v1/demo/scenario` — Sample synthetic heavy-rain scenario payload
- `POST /api/v1/predict/traffic` — Predict traffic congestion % using the trained Traffic Random Forest model
- `POST /api/v1/predict/flood` — Predict flood probability & risk band using the trained Flood ML classifier
- `GET /api/v1/traffic/live` — Fetch live TomTom/HERE traffic flow with automatic simulated fallback + RF forecast
- `POST /api/v1/analyze` — Execute the 9-agent LangGraph workflow (supports custom readings, live weather/traffic, and LLM toggle) and persist the incident
- `GET /api/v1/incidents` — List persisted incidents from SQLite/PostgreSQL
- `POST /api/v1/approvals/{incident_id}` — Record human operator `approved` or `rejected` decision
- `GET /api/v1/knowledge/status` — Inspect FAISS/Chroma RAG index status and indexed documents
- `POST /api/v1/knowledge/search` — Semantic vector search over ingested SOP documents
- `POST /api/v1/knowledge/ingest` — Upload PDF/TXT/MD SOP (max 10 MB) and immediately re-index FAISS

---

## Testing & Verification
Run the full backend test suite from `CITYPULSE-AI`:
```powershell
pytest
```
With FastAPI (`127.0.0.1:8000`) and Next.js (`localhost:3000` or `3001`) running, run the live end-to-end HTTP + CORS verification script:
```powershell
python backend/scripts/verify_e2e.py
```
Verify the frontend TypeScript & production build:
```powershell
cd frontend
npm run build
```

---

## Deployment (Docker Compose)
Build and launch the full stack (FastAPI backend, Next.js frontend, PostgreSQL/PostGIS, Redis):
```powershell
Copy-Item .env.example .env
docker compose up --build -d
```
- Frontend: `http://localhost:3000`
- Backend API & Swagger docs: `http://localhost:8000/docs`
