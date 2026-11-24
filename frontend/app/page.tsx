"use client";

import { useCallback, useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api/v1";

type SystemStatus = {
  status: string;
  models: {
    traffic_model: { loaded: boolean; training_data_type: string };
    flood_model: {
      loaded: boolean;
      selected_algorithm: string;
      training_data_type: string;
      evaluation_metrics?: { accuracy?: number; f1_score?: number; roc_auc?: number };
    };
  };
  rag: {
    active_mode: string;
    faiss_available: boolean;
    embedding_backend: string;
    indexed_documents: number;
    indexed_chunks: number;
    document_sources: string[];
    index_persisted: boolean;
  };
  llm: {
    llm_enabled: boolean;
    provider: string;
    model: string;
    ready: boolean;
  };
  traffic_api: {
    enabled: boolean;
    provider: string;
    api_key_configured: boolean;
  };
};

type AgentFinding = {
  agent: string;
  summary: string;
  reasoning_mode?: string;
  findings?: Record<string, unknown>;
};

type AnalyzeResult = {
  incident_id: string;
  zone: string;
  scenario: string;
  readings: Record<string, number | string>;
  data_provenance?: {
    sensor_readings_source: string;
    traffic_data_origin: string;
    traffic_is_simulated: boolean;
    weather_is_live: boolean;
    ml_training_data: string;
  };
  llm_summary?: {
    llm_used: boolean;
    llm_nodes_count: number;
    reasoning_mode: string;
    provider: string;
    model: string;
  };
  traffic_context?: LiveTrafficResult | null;
  weather_context?: Record<string, unknown> | null;
  risk_assessment: {
    overall_score: number;
    severity: string;
    domain_scores?: { flood: number; traffic: number; environment: number };
    ml_scores?: { traffic_congestion_pct: number; flood_risk_pct: number };
  };
  agent_findings: AgentFinding[];
  response_plan: {
    actions: { priority: string; action: string; rationale?: string; source_citation?: string }[];
    safety_note: string;
    reasoning_mode?: string;
  };
  workflow_trace: string[];
  status: string;
  disclaimer: string;
};

type TrafficPredResult = {
  zone: string;
  predicted_congestion_pct: number;
  traffic_status: string;
  model_info: {
    model_name: string;
    method: string;
    model_available: boolean;
    feature_importances: Record<string, number>;
  };
  data_provenance: {
    input_source: string;
    is_synthetic_training_data: boolean;
    training_dataset: string;
  };
  disclaimer: string;
};

type FloodPredResult = {
  zone: string;
  flood_probability: number;
  flood_risk_pct: number;
  flood_predicted: boolean;
  risk_level: string;
  model_info: {
    model_name: string;
    selected_algorithm: string;
    model_available: boolean;
    evaluation_metrics?: { accuracy?: number; f1_score?: number; roc_auc?: number };
    feature_importances: Record<string, number>;
  };
  data_provenance: {
    input_source: string;
    is_synthetic_training_data: boolean;
    training_dataset: string;
  };
  disclaimer: string;
};

type LiveTrafficResult = {
  zone: string;
  provider: string;
  data_origin: string;
  is_simulated: boolean;
  fallback_reason: string | null;
  timestamp_utc: string;
  coordinates: { latitude: number; longitude: number };
  metrics: {
    current_speed_kmph: number;
    free_flow_speed_kmph: number;
    congestion_pct: number;
    delay_sec?: number;
  };
  ml_prediction?: TrafficPredResult;
  note: string;
};

type RagHit = {
  source: string;
  section?: string;
  chunk_id?: string;
  text: string;
  score: number;
  retrieval_method?: string;
  embedding_backend?: string;
};

type IncidentRow = {
  id: string;
  zone: string;
  scenario: string;
  severity: string;
  risk_score: number;
  status: string;
  summary: string;
  created_at: string;
};

function severityClass(level: string): string {
  const v = level.toLowerCase();
  if (v === "high") return "sev-high";
  if (v === "medium" || v === "moderate") return "sev-medium";
  return "sev-low";
}

function barColor(pct: number): string {
  if (pct >= 70) return "#ff6b7a";
  if (pct >= 40) return "#f5b942";
  return "#44d19e";
}

export default function Home() {
  const [tab, setTab] = useState<"ops" | "ml" | "rag" | "incidents">("ops");
  const [apiStatus, setApiStatus] = useState("Checking API...");
  const [sysStatus, setSysStatus] = useState<SystemStatus | null>(null);

  // Operations workflow state
  const [scenario, setScenario] = useState("heavy_rain");
  const [zone, setZone] = useState("Zone A");
  const [useLiveTraffic, setUseLiveTraffic] = useState(false);
  const [useLiveWeather, setUseLiveWeather] = useState(false);
  const [useLlm, setUseLlm] = useState(false);
  const [customReadings, setCustomReadings] = useState({
    rainfall_mm_hour: 60,
    traffic_density_pct: 75,
    water_level_m: 0.9,
    aqi: 140,
    drainage_capacity_pct: 50,
  });
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Approval state
  const [reviewer, setReviewer] = useState("Ops Officer");
  const [approvalComment, setApprovalComment] = useState("");
  const [approvalMsg, setApprovalMsg] = useState("");

  // Traffic ML predictor state
  const [trafficForm, setTrafficForm] = useState({
    hour: 18,
    day_of_week: 2,
    vehicle_count: 950,
    average_speed: 25,
    road_capacity: 600,
    rainfall_mm: 12,
    special_event: 1,
    zone: "Zone A",
  });
  const [trafficPred, setTrafficPred] = useState<TrafficPredResult | null>(null);
  const [trafficLoading, setTrafficLoading] = useState(false);

  // Flood ML predictor state
  const [floodForm, setFloodForm] = useState({
    rainfall_mm_hour: 85,
    water_level_m: 1.35,
    drainage_capacity_pct: 40,
    previous_24h_rainfall_mm: 120,
    soil_moisture_pct: 78,
    elevation_m: 12,
    zone: "Zone A",
  });
  const [floodPred, setFloodPred] = useState<FloodPredResult | null>(null);
  const [floodLoading, setFloodLoading] = useState(false);

  // Live Traffic feed state
  const [liveZone, setLiveZone] = useState("Zone A");
  const [liveTraffic, setLiveTraffic] = useState<LiveTrafficResult | null>(null);
  const [liveLoading, setLiveLoading] = useState(false);

  // RAG state
  const [ragQuery, setRagQuery] = useState("flood drainage inspection and traffic corridor diversion");
  const [ragHits, setRagHits] = useState<RagHit[]>([]);
  const [ragMeta, setRagMeta] = useState<{ rag_mode: string; embedding_backend: string; indexed_chunks: number } | null>(null);
  const [ragLoading, setRagLoading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");

  // Incidents state
  const [incidents, setIncidents] = useState<IncidentRow[]>([]);
  const [incidentsLoading, setIncidentsLoading] = useState(false);

  const loadSystemStatus = useCallback(async () => {
    try {
      const healthRes = await fetch(`${API}/health`);
      if (!healthRes.ok) {
        setApiStatus(`API error (${healthRes.status})`);
        return;
      }
      const r = await fetch(`${API}/system/status`);
      if (r.ok) {
        const data: SystemStatus = await r.json();
        setSysStatus(data);
        setApiStatus("API connected");
        setUseLlm(Boolean(data.llm.ready));
      } else {
        setApiStatus("API connected");
      }
    } catch {
      setApiStatus("API unavailable — start FastAPI");
    }
  }, []);

  const loadIncidents = useCallback(async () => {
    setIncidentsLoading(true);
    try {
      const r = await fetch(`${API}/incidents?limit=25`);
      if (r.ok) setIncidents(await r.json());
    } catch {
      // Handled gracefully when FastAPI is not reachable yet
    } finally {
      setIncidentsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSystemStatus();
    loadIncidents();
  }, [loadSystemStatus, loadIncidents]);

  async function runAnalysis() {
    setLoading(true);
    setError("");
    setApprovalMsg("");
    try {
      const payload: Record<string, unknown> = {
        scenario,
        zone,
        use_live_weather: useLiveWeather,
        use_live_traffic: useLiveTraffic,
        use_llm: useLlm,
      };
      if (scenario === "custom") {
        payload.readings = { ...customReadings, source: "user_supplied" };
      }
      const r = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) throw new Error(await r.text());
      const data: AnalyzeResult = await r.json();
      setResult(data);
      loadIncidents();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis request failed");
    } finally {
      setLoading(false);
    }
  }

  async function submitApproval(incidentId: string, decision: "approved" | "rejected") {
    setError("");
    setApprovalMsg("");
    try {
      const r = await fetch(`${API}/approvals/${incidentId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          decision,
          reviewer: reviewer.trim().length >= 2 ? reviewer.trim() : "Ops Officer",
          comment: approvalComment,
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setApprovalMsg(`Incident ${data.incident_id.slice(0, 8)} marked as ${data.status.toUpperCase()}.`);
      if (result && result.incident_id === incidentId) {
        setResult({ ...result, status: data.status });
      }
      loadIncidents();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Approval failed");
    }
  }

  async function runTrafficPredict() {
    setTrafficLoading(true);
    setError("");
    try {
      const r = await fetch(`${API}/predict/traffic`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...trafficForm, data_source: "user_supplied" }),
      });
      if (!r.ok) throw new Error(await r.text());
      setTrafficPred(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Traffic prediction failed");
    } finally {
      setTrafficLoading(false);
    }
  }

  async function runFloodPredict() {
    setFloodLoading(true);
    setError("");
    try {
      const r = await fetch(`${API}/predict/flood`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...floodForm, data_source: "user_supplied" }),
      });
      if (!r.ok) throw new Error(await r.text());
      setFloodPred(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Flood prediction failed");
    } finally {
      setFloodLoading(false);
    }
  }

  async function fetchLiveTraffic() {
    setLiveLoading(true);
    setError("");
    try {
      const r = await fetch(`${API}/traffic/live?zone=${encodeURIComponent(liveZone)}&use_live=true`);
      if (!r.ok) throw new Error(await r.text());
      setLiveTraffic(await r.json());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Live traffic fetch failed");
    } finally {
      setLiveLoading(false);
    }
  }

  async function runRagSearch() {
    if (ragQuery.trim().length < 3) return;
    setRagLoading(true);
    setError("");
    try {
      const r = await fetch(`${API}/knowledge/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: ragQuery, top_k: 4 }),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setRagHits(data.results || []);
      setRagMeta({
        rag_mode: data.rag_mode,
        embedding_backend: data.embedding_backend,
        indexed_chunks: data.indexed_chunks,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Knowledge search failed");
    } finally {
      setRagLoading(false);
    }
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadMsg("");
    setError("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API}/knowledge/ingest`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setUploadMsg(`Ingested ${data.filename} (${data.characters} chars) — indexed ${data.indexed_chunks} chunks in ${data.rag_mode.toUpperCase()}.`);
      loadSystemStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      e.target.value = "";
    }
  }

  const commFindings = result?.agent_findings.find((f) => f.agent === "communication_reporting_agent")?.findings as
    | { executive_briefing?: string; citizen_advisory_draft?: string }
    | undefined;

  return (
    <main className="shell">
      <header className="top">
        <div>
          <div className="brand">
            CITYPULSE <span style={{ color: "#38d9b4" }}>AI</span>
          </div>
          <div className="muted">Autonomous Smart City Intelligence &amp; Operations Platform</div>
        </div>
        <div className="pill-row">
          <span className={`pill ${apiStatus.includes("connected") ? "" : "pill-danger"}`}>{apiStatus}</span>
          {sysStatus && (
            <>
              <span className="pill pill-info">
                Traffic ML: {sysStatus.models.traffic_model.loaded ? "RF Ready" : "Fallback"}
              </span>
              <span className="pill pill-info">
                Flood ML: {sysStatus.models.flood_model.loaded ? "Ready" : "Fallback"}
              </span>
              <span className="pill">
                RAG: {sysStatus.rag.active_mode.toUpperCase()} ({sysStatus.rag.indexed_chunks} chunks)
              </span>
              <span className={`pill ${sysStatus.llm.ready ? "" : "pill-warn"}`}>
                LLM: {sysStatus.llm.ready ? `${sysStatus.llm.provider}/${sysStatus.llm.model}` : "Deterministic Fallback"}
              </span>
            </>
          )}
        </div>
      </header>

      <nav className="tabs">
        <button className={`tab-btn ${tab === "ops" ? "active" : ""}`} onClick={() => setTab("ops")}>
          Multi-Agent Operations
        </button>
        <button className={`tab-btn ${tab === "ml" ? "active" : ""}`} onClick={() => setTab("ml")}>
          ML Predictors &amp; Live Traffic
        </button>
        <button className={`tab-btn ${tab === "rag" ? "active" : ""}`} onClick={() => setTab("rag")}>
          FAISS Knowledge RAG
        </button>
        <button
          className={`tab-btn ${tab === "incidents" ? "active" : ""}`}
          onClick={() => {
            setTab("incidents");
            loadIncidents();
          }}
        >
          Incidents &amp; Approvals ({incidents.length})
        </button>
      </nav>

      {error && <div className="error">{error}</div>}
      {approvalMsg && <div className="success">{approvalMsg}</div>}

      {tab === "ops" && (
        <>
          <section className="hero">
            <span className="pill">LANGGRAPH 9-AGENT DECISION SUPPORT · HUMAN-IN-THE-LOOP</span>
            <h1>City operations, connected.</h1>
            <p className="muted">
              Combines trained Traffic &amp; Flood ML models, real-time traffic/weather connectors with graceful
              fallback, FAISS embedding RAG, and optional LLM multi-agent reasoning.
            </p>
            <div className="controls">
              <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
                <option value="heavy_rain">Heavy rainfall (Synthetic Demo)</option>
                <option value="traffic_peak">Traffic peak (Synthetic Demo)</option>
                <option value="poor_air_quality">Poor air quality (Synthetic Demo)</option>
                <option value="normal">Normal conditions (Synthetic Demo)</option>
                <option value="custom">Custom sensor readings</option>
              </select>
              <select value={zone} onChange={(e) => setZone(e.target.value)}>
                <option>Zone A</option>
                <option>Zone B</option>
                <option>Zone C</option>
              </select>
              <button onClick={runAnalysis} disabled={loading}>
                {loading ? "Running 9-agent workflow..." : "Run multi-agent analysis"}
              </button>
            </div>
            <div className="toggle-row">
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={useLiveTraffic}
                  onChange={(e) => setUseLiveTraffic(e.target.checked)}
                />
                Use live traffic API (simulated fallback if unconfigured)
              </label>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={useLiveWeather}
                  onChange={(e) => setUseLiveWeather(e.target.checked)}
                />
                Use live weather (Open-Meteo)
              </label>
              <label className="toggle">
                <input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
                Use LLM agent reasoning (falls back deterministically if no key)
              </label>
            </div>

            {scenario === "custom" && (
              <div className="grid-3" style={{ marginTop: 14 }}>
                <div className="field">
                  <label>Rainfall (mm/hr)</label>
                  <input
                    type="number"
                    value={customReadings.rainfall_mm_hour}
                    onChange={(e) =>
                      setCustomReadings({ ...customReadings, rainfall_mm_hour: Number(e.target.value) })
                    }
                  />
                </div>
                <div className="field">
                  <label>Traffic Density (%)</label>
                  <input
                    type="number"
                    value={customReadings.traffic_density_pct}
                    onChange={(e) =>
                      setCustomReadings({ ...customReadings, traffic_density_pct: Number(e.target.value) })
                    }
                  />
                </div>
                <div className="field">
                  <label>Water Level (m)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={customReadings.water_level_m}
                    onChange={(e) =>
                      setCustomReadings({ ...customReadings, water_level_m: Number(e.target.value) })
                    }
                  />
                </div>
                <div className="field">
                  <label>AQI</label>
                  <input
                    type="number"
                    value={customReadings.aqi}
                    onChange={(e) => setCustomReadings({ ...customReadings, aqi: Number(e.target.value) })}
                  />
                </div>
                <div className="field">
                  <label>Drainage Capacity (%)</label>
                  <input
                    type="number"
                    value={customReadings.drainage_capacity_pct}
                    onChange={(e) =>
                      setCustomReadings({ ...customReadings, drainage_capacity_pct: Number(e.target.value) })
                    }
                  />
                </div>
              </div>
            )}
          </section>

          {result && (
            <>
              <div className="provenance-bar">
                <strong style={{ fontSize: 13 }}>Data Provenance:</strong>
                <span className="pill pill-warn">
                  Sensors: {result.data_provenance?.sensor_readings_source || "simulated_demo"}
                </span>
                <span className={`pill ${result.data_provenance?.traffic_is_simulated ? "pill-warn" : ""}`}>
                  Traffic: {result.data_provenance?.traffic_data_origin || "simulated_demo"}
                </span>
                <span className={`pill ${result.data_provenance?.weather_is_live ? "" : "pill-warn"}`}>
                  Weather: {result.data_provenance?.weather_is_live ? "Open-Meteo Live" : "Demo/Off"}
                </span>
                <span className="pill pill-info">ML Training Data: Synthetic</span>
                <span className={`pill ${result.llm_summary?.llm_used ? "" : "pill-warn"}`}>
                  Reasoning: {result.llm_summary?.reasoning_mode || "deterministic_fallback"}
                </span>
              </div>

              <section className="section">
                <h2>
                  Incident Intelligence · {result.zone} ({result.status.replaceAll("_", " ")})
                </h2>
                <div className="grid">
                  <div className="card">
                    <div className="muted">Illustrative Risk Score</div>
                    <div className={`metric ${severityClass(result.risk_assessment.severity)}`}>
                      {result.risk_assessment.overall_score}/100
                    </div>
                    <div className="muted">Severity: {result.risk_assessment.severity.toUpperCase()}</div>
                  </div>
                  <div className="card">
                    <div className="muted">Traffic RF ML Prediction</div>
                    <div className="metric">
                      {result.risk_assessment.ml_scores?.traffic_congestion_pct ?? "—"}%
                    </div>
                    <div className="muted">Random Forest (synthetic)</div>
                  </div>
                  <div className="card">
                    <div className="muted">Flood ML Probability</div>
                    <div className="metric">
                      {result.risk_assessment.ml_scores?.flood_risk_pct ?? "—"}%
                    </div>
                    <div className="muted">Flood Classifier (synthetic)</div>
                  </div>
                  <div className="card">
                    <div className="muted">Incident ID</div>
                    <div className="metric-sm" style={{ overflowWrap: "anywhere" }}>
                      {result.incident_id.slice(0, 18)}…
                    </div>
                    <div className="muted">Status: {result.status}</div>
                  </div>
                </div>
              </section>

              {commFindings?.executive_briefing && (
                <section className="section">
                  <div className="card">
                    <div className="finding-head">
                      <strong>Executive Operations Briefing</strong>
                      <span className="pill pill-warn">DRAFT ONLY</span>
                    </div>
                    <p style={{ margin: "10px 0 6px" }}>{commFindings.executive_briefing}</p>
                    {commFindings.citizen_advisory_draft && (
                      <p className="muted" style={{ margin: 0 }}>
                        <b>Citizen Advisory Draft:</b> {commFindings.citizen_advisory_draft}
                      </p>
                    )}
                  </div>
                </section>
              )}

              <section className="section">
                <h2>9-Agent Specialist Findings</h2>
                <div className="findings">
                  {result.agent_findings.map((f) => (
                    <div className="finding" key={f.agent}>
                      <div className="finding-head">
                        <b style={{ textTransform: "capitalize" }}>{f.agent.replaceAll("_", " ")}</b>
                        {f.reasoning_mode && (
                          <span className={`pill ${f.reasoning_mode === "llm" ? "" : "pill-info"}`}>
                            {f.reasoning_mode}
                          </span>
                        )}
                      </div>
                      <p className="muted" style={{ margin: 0 }}>
                        {f.summary}
                      </p>
                    </div>
                  ))}
                </div>
              </section>

              <section className="section">
                <h2>Proposed Response Plan &amp; Human Approval</h2>
                <div className="card">
                  {result.response_plan.actions.map((a, idx) => (
                    <div className="action-item" key={idx}>
                      <div>
                        <span className={`pill ${a.priority === "high" ? "pill-danger" : "pill-warn"}`}>
                          {a.priority.toUpperCase()}
                        </span>{" "}
                        <strong>{a.action}</strong>
                      </div>
                      {a.rationale && (
                        <div className="muted" style={{ marginTop: 4 }}>
                          {a.rationale}
                        </div>
                      )}
                      {a.source_citation && (
                        <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
                          Source: <code>{a.source_citation}</code>
                        </div>
                      )}
                    </div>
                  ))}
                  <p className="muted">{result.response_plan.safety_note}</p>

                  <div className="approval-box">
                    <div className="field">
                      <label>Authorised Reviewer</label>
                      <input
                        type="text"
                        value={reviewer}
                        onChange={(e) => setReviewer(e.target.value)}
                        placeholder="Operations Officer"
                      />
                    </div>
                    <div className="field">
                      <label>Review Comment</label>
                      <input
                        type="text"
                        value={approvalComment}
                        onChange={(e) => setApprovalComment(e.target.value)}
                        placeholder="Verified against official sources..."
                      />
                    </div>
                    <button onClick={() => submitApproval(result.incident_id, "approved")}>Approve Plan</button>
                    <button className="btn-danger" onClick={() => submitApproval(result.incident_id, "rejected")}>
                      Reject Plan
                    </button>
                  </div>
                </div>
              </section>

              <section className="section">
                <h2>LangGraph Workflow Trace</h2>
                <pre>{result.workflow_trace.join("\n")}</pre>
                <p className="muted">{result.disclaimer}</p>
              </section>
            </>
          )}
        </>
      )}

      {tab === "ml" && (
        <>
          <div className="grid-2">
            <section className="card">
              <div className="finding-head">
                <div>
                  <h2 style={{ margin: 0 }}>Traffic Congestion Predictor</h2>
                  <p className="muted" style={{ margin: "4px 0 0" }}>
                    Trained Random Forest Regressor (<code>traffic_model.joblib</code>)
                  </p>
                </div>
                <span className="pill pill-warn">Synthetic ML</span>
              </div>

              <div className="form-grid">
                <div className="field">
                  <label>Hour (0–23)</label>
                  <input
                    type="number"
                    min={0}
                    max={23}
                    value={trafficForm.hour}
                    onChange={(e) => setTrafficForm({ ...trafficForm, hour: Number(e.target.value) })}
                  />
                </div>
                <div className="field">
                  <label>Day of Week (0=Mon..6=Sun)</label>
                  <input
                    type="number"
                    min={0}
                    max={6}
                    value={trafficForm.day_of_week}
                    onChange={(e) => setTrafficForm({ ...trafficForm, day_of_week: Number(e.target.value) })}
                  />
                </div>
                <div className="field">
                  <label>Vehicle Count</label>
                  <input
                    type="number"
                    min={0}
                    value={trafficForm.vehicle_count}
                    onChange={(e) => setTrafficForm({ ...trafficForm, vehicle_count: Number(e.target.value) })}
                  />
                </div>
                <div className="field">
                  <label>Average Speed (km/h)</label>
                  <input
                    type="number"
                    min={0}
                    value={trafficForm.average_speed}
                    onChange={(e) => setTrafficForm({ ...trafficForm, average_speed: Number(e.target.value) })}
                  />
                </div>
                <div className="field">
                  <label>Road Capacity</label>
                  <input
                    type="number"
                    min={1}
                    value={trafficForm.road_capacity}
                    onChange={(e) => setTrafficForm({ ...trafficForm, road_capacity: Number(e.target.value) })}
                  />
                </div>
                <div className="field">
                  <label>Rainfall (mm)</label>
                  <input
                    type="number"
                    min={0}
                    step="0.5"
                    value={trafficForm.rainfall_mm}
                    onChange={(e) => setTrafficForm({ ...trafficForm, rainfall_mm: Number(e.target.value) })}
                  />
                </div>
                <div className="field">
                  <label>Special Event (0 or 1)</label>
                  <select
                    value={trafficForm.special_event}
                    onChange={(e) => setTrafficForm({ ...trafficForm, special_event: Number(e.target.value) })}
                  >
                    <option value={0}>0 — None</option>
                    <option value={1}>1 — Special Event</option>
                  </select>
                </div>
                <div className="field">
                  <label>Zone</label>
                  <select
                    value={trafficForm.zone}
                    onChange={(e) => setTrafficForm({ ...trafficForm, zone: e.target.value })}
                  >
                    <option>Zone A</option>
                    <option>Zone B</option>
                    <option>Zone C</option>
                  </select>
                </div>
              </div>

              <button onClick={runTrafficPredict} disabled={trafficLoading}>
                {trafficLoading ? "Predicting..." : "Predict Traffic Congestion"}
              </button>

              {trafficPred && (
                <div style={{ marginTop: 16 }}>
                  <div className="finding-head">
                    <span className={`metric ${severityClass(trafficPred.traffic_status)}`}>
                      {trafficPred.predicted_congestion_pct}%
                    </span>
                    <span className="pill">{trafficPred.traffic_status}</span>
                  </div>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{
                        width: `${trafficPred.predicted_congestion_pct}%`,
                        background: barColor(trafficPred.predicted_congestion_pct),
                      }}
                    />
                  </div>
                  <p className="muted" style={{ fontSize: 12 }}>
                    {trafficPred.disclaimer}
                  </p>
                </div>
              )}
            </section>

            <section className="card">
              <div className="finding-head">
                <div>
                  <h2 style={{ margin: 0 }}>Flood Risk Predictor</h2>
                  <p className="muted" style={{ margin: "4px 0 0" }}>
                    Trained Flood Classifier (<code>flood_model.joblib</code>)
                  </p>
                </div>
                <span className="pill pill-warn">Synthetic ML</span>
              </div>

              <div className="form-grid">
                <div className="field">
                  <label>Rainfall (mm/hr)</label>
                  <input
                    type="number"
                    min={0}
                    value={floodForm.rainfall_mm_hour}
                    onChange={(e) => setFloodForm({ ...floodForm, rainfall_mm_hour: Number(e.target.value) })}
                  />
                </div>
                <div className="field">
                  <label>Water Level (m)</label>
                  <input
                    type="number"
                    min={0}
                    step="0.05"
                    value={floodForm.water_level_m}
                    onChange={(e) => setFloodForm({ ...floodForm, water_level_m: Number(e.target.value) })}
                  />
                </div>
                <div className="field">
                  <label>Drainage Capacity (%)</label>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    value={floodForm.drainage_capacity_pct}
                    onChange={(e) => setFloodForm({ ...floodForm, drainage_capacity_pct: Number(e.target.value) })}
                  />
                </div>
                <div className="field">
                  <label>Previous 24h Rainfall (mm)</label>
                  <input
                    type="number"
                    min={0}
                    value={floodForm.previous_24h_rainfall_mm}
                    onChange={(e) =>
                      setFloodForm({ ...floodForm, previous_24h_rainfall_mm: Number(e.target.value) })
                    }
                  />
                </div>
                <div className="field">
                  <label>Soil Moisture (%)</label>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    value={floodForm.soil_moisture_pct}
                    onChange={(e) => setFloodForm({ ...floodForm, soil_moisture_pct: Number(e.target.value) })}
                  />
                </div>
                <div className="field">
                  <label>Zone Elevation (m)</label>
                  <input
                    type="number"
                    min={0}
                    value={floodForm.elevation_m}
                    onChange={(e) => setFloodForm({ ...floodForm, elevation_m: Number(e.target.value) })}
                  />
                </div>
              </div>

              <button onClick={runFloodPredict} disabled={floodLoading}>
                {floodLoading ? "Predicting..." : "Predict Flood Risk"}
              </button>

              {floodPred && (
                <div style={{ marginTop: 16 }}>
                  <div className="finding-head">
                    <span className={`metric ${severityClass(floodPred.risk_level)}`}>
                      {floodPred.flood_risk_pct}%
                    </span>
                    <span className="pill">{floodPred.risk_level} RISK</span>
                  </div>
                  <div className="bar-track">
                    <div
                      className="bar-fill"
                      style={{
                        width: `${floodPred.flood_risk_pct}%`,
                        background: barColor(floodPred.flood_risk_pct),
                      }}
                    />
                  </div>
                  <p className="muted" style={{ fontSize: 12 }}>
                    Algorithm: <b>{floodPred.model_info.selected_algorithm}</b> · Synthetic Test ROC-AUC:{" "}
                    <b>{floodPred.model_info.evaluation_metrics?.roc_auc ?? "N/A"}</b> · Accuracy:{" "}
                    <b>{floodPred.model_info.evaluation_metrics?.accuracy ?? "N/A"}</b>
                  </p>
                  <p className="muted" style={{ fontSize: 12 }}>
                    {floodPred.disclaimer}
                  </p>
                </div>
              )}
            </section>
          </div>

          <section className="card section">
            <div className="finding-head">
              <div>
                <h2 style={{ margin: 0 }}>Real-Time Traffic API Feed (with Simulated Fallback)</h2>
                <p className="muted" style={{ margin: "4px 0 0" }}>
                  Connects to TomTom or HERE Traffic Flow API when configured via environment variables, or gracefully
                  falls back to time-of-day simulated telemetry.
                </p>
              </div>
              <div className="controls" style={{ marginTop: 0 }}>
                <select value={liveZone} onChange={(e) => setLiveZone(e.target.value)}>
                  <option>Zone A</option>
                  <option>Zone B</option>
                  <option>Zone C</option>
                </select>
                <button onClick={fetchLiveTraffic} disabled={liveLoading}>
                  {liveLoading ? "Fetching..." : "Fetch Traffic Snapshot"}
                </button>
              </div>
            </div>

            {liveTraffic && (
              <div style={{ marginTop: 16 }}>
                <div className="provenance-bar">
                  <span className={`pill ${liveTraffic.is_simulated ? "pill-warn" : ""}`}>
                    {liveTraffic.is_simulated ? "SIMULATED FALLBACK" : "LIVE TRAFFIC API"}
                  </span>
                  <span className="pill pill-info">Provider: {liveTraffic.provider}</span>
                  <span className="muted" style={{ fontSize: 12 }}>
                    Coords: {liveTraffic.coordinates.latitude}, {liveTraffic.coordinates.longitude}
                  </span>
                </div>
                <div className="grid">
                  <div className="card">
                    <div className="muted">Current Speed</div>
                    <div className="metric">{liveTraffic.metrics.current_speed_kmph} km/h</div>
                  </div>
                  <div className="card">
                    <div className="muted">Free-Flow Speed</div>
                    <div className="metric">{liveTraffic.metrics.free_flow_speed_kmph} km/h</div>
                  </div>
                  <div className="card">
                    <div className="muted">Observed Congestion</div>
                    <div className="metric">{liveTraffic.metrics.congestion_pct}%</div>
                  </div>
                  <div className="card">
                    <div className="muted">Traffic RF ML Forecast</div>
                    <div className="metric">
                      {liveTraffic.ml_prediction?.predicted_congestion_pct ?? "—"}%
                    </div>
                  </div>
                </div>
                {liveTraffic.fallback_reason && (
                  <p className="muted" style={{ fontSize: 13 }}>
                    <b>Fallback reason:</b> {liveTraffic.fallback_reason}
                  </p>
                )}
              </div>
            )}
          </section>
        </>
      )}

      {tab === "rag" && (
        <section className="card">
          <div className="finding-head">
            <div>
              <h2 style={{ margin: 0 }}>FAISS Embedding-Based Knowledge Base (RAG)</h2>
              <p className="muted" style={{ margin: "4px 0 0" }}>
                Semantic retrieval over city operating procedures with persisted FAISS vector indexing.
              </p>
            </div>
            {sysStatus && (
              <div className="pill-row">
                <span className="pill">{sysStatus.rag.active_mode.toUpperCase()}</span>
                <span className="pill pill-info">{sysStatus.rag.embedding_backend}</span>
                <span className="pill">
                  {sysStatus.rag.indexed_documents} docs · {sysStatus.rag.indexed_chunks} chunks
                </span>
              </div>
            )}
          </div>

          <div className="controls">
            <input
              type="text"
              style={{ flex: 1, minWidth: 260 }}
              value={ragQuery}
              onChange={(e) => setRagQuery(e.target.value)}
              placeholder="Ask about flood checklists, corridor diversion, AQI advisories, or storm drains..."
            />
            <button onClick={runRagSearch} disabled={ragLoading}>
              {ragLoading ? "Searching..." : "Semantic Search"}
            </button>
          </div>

          <div style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <label className="muted" style={{ fontWeight: 600 }}>
              Ingest new SOP document (PDF / TXT / MD):
            </label>
            <input type="file" accept=".pdf,.txt,.md" onChange={handleFileUpload} />
          </div>
          {uploadMsg && <div className="success">{uploadMsg}</div>}

          {ragMeta && (
            <p className="muted" style={{ marginTop: 14 }}>
              Retrieved via <b>{ragMeta.rag_mode.toUpperCase()}</b> ({ragMeta.embedding_backend}) across{" "}
              <b>{ragMeta.indexed_chunks}</b> indexed chunks.
            </p>
          )}

          <div className="section">
            {ragHits.map((hit, i) => (
              <div className="action-item" key={hit.chunk_id || i}>
                <div className="finding-head">
                  <strong>
                    {hit.source} {hit.section ? `· ${hit.section}` : ""}
                  </strong>
                  <span className="pill">Score: {hit.score}</span>
                </div>
                <p className="muted" style={{ whiteSpace: "pre-wrap", marginBottom: 0 }}>
                  {hit.text}
                </p>
              </div>
            ))}
          </div>
        </section>
      )}

      {tab === "incidents" && (
        <section className="card">
          <div className="finding-head" style={{ marginBottom: 12 }}>
            <div>
              <h2 style={{ margin: 0 }}>Persisted Incident Log &amp; Human Approvals</h2>
              <p className="muted" style={{ margin: "4px 0 0" }}>
                Stored in SQLite (or PostgreSQL via DATABASE_URL).
              </p>
            </div>
            <button className="btn-secondary" onClick={loadIncidents} disabled={incidentsLoading}>
              {incidentsLoading ? "Refreshing..." : "Refresh Log"}
            </button>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Zone</th>
                  <th>Scenario</th>
                  <th>Risk</th>
                  <th>Status</th>
                  <th>Summary</th>
                  <th>Review Action</th>
                </tr>
              </thead>
              <tbody>
                {incidents.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="muted">
                      No incidents recorded yet. Run an analysis in the Multi-Agent Operations tab.
                    </td>
                  </tr>
                ) : (
                  incidents.map((row) => (
                    <tr key={row.id}>
                      <td>{row.zone}</td>
                      <td>{row.scenario}</td>
                      <td className={severityClass(row.severity)}>
                        <b>
                          {row.risk_score}/100 ({row.severity.toUpperCase()})
                        </b>
                      </td>
                      <td>
                        <span
                          className={`pill ${
                            row.status === "approved"
                              ? ""
                              : row.status === "rejected"
                                ? "pill-danger"
                                : "pill-warn"
                          }`}
                        >
                          {row.status}
                        </span>
                      </td>
                      <td style={{ maxWidth: 420 }}>{row.summary}</td>
                      <td>
                        {row.status === "awaiting_review" ? (
                          <div style={{ display: "flex", gap: 6 }}>
                            <button onClick={() => submitApproval(row.id, "approved")}>Approve</button>
                            <button className="btn-danger" onClick={() => submitApproval(row.id, "rejected")}>
                              Reject
                            </button>
                          </div>
                        ) : (
                          <span className="muted">Reviewed</span>
                        )}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <p className="muted section" style={{ fontSize: 12 }}>
        Prototype decision-support platform only. Synthetic ML training datasets and simulated fallback readings are
        explicitly labeled and must not be used for real-world emergency dispatch or public flood/traffic warnings.
      </p>
    </main>
  );
}
