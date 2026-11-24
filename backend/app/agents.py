from __future__ import annotations

from backend.app.llm import (
    CommunicationLLMOutput,
    ResponsePlanLLMOutput,
    RiskSynthesisLLMOutput,
    SupervisorLLMOutput,
    call_llm_json,
)
from backend.app.ml import (
    assess_risk,
    derive_flood_features_from_readings,
    derive_traffic_features_from_readings,
    predict_flood_risk,
    predict_traffic_congestion,
)
from backend.app.rag import search_knowledge


def supervisor_agent(s: dict) -> dict:
    readings = s["readings"]
    use_llm = s.get("use_llm")
    llm_out, llm_meta = call_llm_json(
        system_prompt=(
            "You are the City Supervisor Agent in CITYPULSE AI, a smart-city decision-support prototype. "
            "Analyze the incoming incident scenario, zone, and sensor readings, then return JSON with keys: "
            "'priority_domains' (list of strings from traffic, flood, environment, infrastructure), "
            "'coordination_brief' (concise 1-2 sentence triage summary), and "
            "'data_quality_watchouts' (list of short caveats noting synthetic vs live data). "
            "Never claim autonomous dispatch or real emergency authority."
        ),
        payload={
            "scenario": s["scenario"],
            "zone": s["zone"],
            "readings": readings,
            "has_live_traffic": bool(s.get("traffic_context") and not s["traffic_context"].get("is_simulated", True)),
            "has_live_weather": bool(s.get("weather_context") and s["weather_context"].get("enabled")),
        },
        schema_cls=SupervisorLLMOutput,
        use_llm=use_llm,
    )

    if llm_out is not None:
        priority_domains = llm_out.priority_domains or ["traffic", "flood", "environment", "infrastructure"]
        brief = llm_out.coordination_brief
        watchouts = llm_out.data_quality_watchouts
    else:
        priority_domains = []
        if float(readings.get("rainfall_mm_hour", 0)) >= 30 or float(readings.get("water_level_m", 0)) >= 0.5:
            priority_domains.append("flood")
        if float(readings.get("traffic_density_pct", 0)) >= 50:
            priority_domains.append("traffic")
        if float(readings.get("aqi", 0)) >= 100:
            priority_domains.append("environment")
        if float(readings.get("drainage_capacity_pct", 100)) < 60:
            priority_domains.append("infrastructure")
        if not priority_domains:
            priority_domains = ["traffic", "flood", "environment", "infrastructure"]
        brief = (
            f"Supervisor received '{s['scenario']}' event for {s['zone']}; "
            f"prioritizing {', '.join(priority_domains)} specialist review."
        )
        watchouts = [f"Primary sensor source: {readings.get('source', 'simulated_demo')}."]

    return {
        "supervisor_finding": {
            "agent": "city_supervisor_agent",
            "summary": brief,
            "reasoning_mode": llm_meta["reasoning_mode"],
            "findings": {
                "priority_domains": priority_domains,
                "data_quality_watchouts": watchouts,
                "llm_meta": llm_meta,
            },
        }
    }


def traffic_agent(s: dict) -> dict:
    density = s["readings"]["traffic_density_pct"]
    traffic_ctx = s.get("traffic_context")
    explicit_features = s.get("traffic_features")
    if explicit_features:
        data_source = explicit_features.get("data_source", "user_supplied")
        ml_result = predict_traffic_congestion(explicit_features, data_source=data_source, zone=s["zone"])
    elif traffic_ctx and traffic_ctx.get("ml_features"):
        ml_result = predict_traffic_congestion(
            traffic_ctx["ml_features"],
            data_source=traffic_ctx.get("data_origin", "simulated_fallback"),
            zone=s["zone"],
        )
    else:
        derived, derived_source = derive_traffic_features_from_readings(s["readings"], s.get("scenario", "normal"))
        ml_result = predict_traffic_congestion(derived, data_source=derived_source, zone=s["zone"])
    predicted_pct = ml_result["predicted_congestion_pct"]
    effective_density = max(float(density), predicted_pct)
    band = "high" if effective_density >= 70 or ml_result["traffic_status"] == "HIGH" else "moderate" if effective_density >= 40 else "low"
    origin = traffic_ctx["data_origin"] if traffic_ctx else ml_result["data_provenance"]["input_source"]
    return {
        "traffic_finding": {
            "agent": "traffic_intelligence_agent",
            "summary": f"Traffic density is {density}% ({origin}); Random Forest ML predicts {predicted_pct:.1f}% congestion ({ml_result['traffic_status']}).",
            "reasoning_mode": "ml_model",
            "findings": {
                "traffic_density_pct": density,
                "congestion_level": band,
                "data_origin": origin,
                "live_traffic_context": traffic_ctx,
                "ml_prediction": ml_result,
            },
        }
    }


def flood_agent(s: dict) -> dict:
    r = s["readings"]
    explicit_features = s.get("flood_features")
    if explicit_features:
        data_source = explicit_features.get("data_source", "user_supplied")
        ml_result = predict_flood_risk(explicit_features, data_source=data_source, zone=s["zone"])
    else:
        derived, derived_source = derive_flood_features_from_readings(r, zone=s["zone"])
        ml_result = predict_flood_risk(derived, data_source=derived_source, zone=s["zone"])
    rule_band = "high" if r["rainfall_mm_hour"] >= 70 or r["water_level_m"] >= 1 else "moderate" if r["rainfall_mm_hour"] >= 30 or r["water_level_m"] >= 0.5 else "low"
    ml_level = ml_result["risk_level"].lower()
    rank = {"low": 0, "moderate": 1, "high": 2}
    band = rule_band if rank[rule_band] >= rank[ml_level] else ml_level
    return {
        "flood_finding": {
            "agent": "weather_flood_agent",
            "summary": f"Flood indicator is {band}; Flood ML estimates {ml_result['flood_risk_pct']:.1f}% risk ({ml_result['risk_level']}).",
            "reasoning_mode": "ml_model",
            "findings": {
                "rainfall_mm_hour": r["rainfall_mm_hour"],
                "water_level_m": r["water_level_m"],
                "indicator_level": band,
                "rule_indicator_level": rule_band,
                "ml_prediction": ml_result,
            },
        }
    }


def environment_agent(s: dict) -> dict:
    aqi = s["readings"]["aqi"]
    band = "high" if aqi >= 150 else "moderate" if aqi >= 100 else "lower"
    return {
        "environment_finding": {
            "agent": "environmental_agent",
            "summary": f"Air-quality indicator is {band} (AQI {aqi}).",
            "reasoning_mode": "deterministic_rules",
            "findings": {"aqi": aqi, "indicator_band": band},
        }
    }


def infrastructure_agent(s: dict) -> dict:
    drainage = s["readings"]["drainage_capacity_pct"]
    flag = "needs_inspection" if drainage < 40 else "monitor"
    return {
        "infrastructure_finding": {
            "agent": "infrastructure_agent",
            "summary": f"Drainage capacity indicator is {drainage}% ({flag.replace('_', ' ')}).",
            "reasoning_mode": "deterministic_rules",
            "findings": {"drainage_capacity_pct": drainage, "maintenance_flag": flag},
        }
    }


def risk_agent(s: dict) -> dict:
    risk = assess_risk(s["readings"])
    traffic_ml_pct = s["traffic_finding"]["findings"]["ml_prediction"]["predicted_congestion_pct"]
    flood_ml_pct = s["flood_finding"]["findings"]["ml_prediction"]["flood_risk_pct"]
    risk["ml_scores"] = {
        "traffic_congestion_pct": traffic_ml_pct,
        "flood_risk_pct": flood_ml_pct,
    }

    baseline_relationships: list[str] = []
    if s["flood_finding"]["findings"]["indicator_level"] == "high" and s["traffic_finding"]["findings"]["congestion_level"] == "high":
        baseline_relationships.append("High flood risk and high traffic congestion overlap in this zone, creating cascading corridor disruption.")
    if s["infrastructure_finding"]["findings"]["maintenance_flag"] == "needs_inspection":
        baseline_relationships.append("Low drainage capacity amplifies surface flooding risk and suggests immediate infrastructure inspection.")
    if s["environment_finding"]["findings"]["indicator_band"] == "high" and s["traffic_finding"]["findings"]["congestion_level"] == "high":
        baseline_relationships.append("Elevated traffic congestion may worsen localized air-quality conditions.")

    llm_out, llm_meta = call_llm_json(
        system_prompt=(
            "You are the Risk Assessment Agent in CITYPULSE AI. Synthesize multi-domain sensor, Traffic ML, and Flood ML findings. "
            "Return JSON with keys: 'cross_domain_relationships' (list of concrete cross-domain interaction statements), "
            "'cascading_risk_narrative' (concise 2-sentence explanation of combined risk), and "
            "'uncertainty_note' (short caveat acknowledging synthetic ML training data or simulated sensors)."
        ),
        payload={
            "zone": s["zone"],
            "scenario": s["scenario"],
            "rule_risk": risk,
            "traffic_summary": s["traffic_finding"]["summary"],
            "flood_summary": s["flood_finding"]["summary"],
            "environment_summary": s["environment_finding"]["summary"],
            "infrastructure_summary": s["infrastructure_finding"]["summary"],
            "baseline_relationships": baseline_relationships,
        },
        schema_cls=RiskSynthesisLLMOutput,
        use_llm=s.get("use_llm"),
    )

    if llm_out is not None:
        relationships = llm_out.cross_domain_relationships or baseline_relationships
        narrative = llm_out.cascading_risk_narrative
        uncertainty_note = llm_out.uncertainty_note
        summary = f"Combined score {risk['overall_score']}/100 ({risk['severity']}). {narrative}"
    else:
        relationships = baseline_relationships
        narrative = (
            f"Rule-based overall score is {risk['overall_score']}/100 ({risk['severity']}); "
            f"ML estimates {traffic_ml_pct:.1f}% traffic congestion and {flood_ml_pct:.1f}% flood risk."
        )
        uncertainty_note = risk["limitation"]
        summary = f"Illustrative combined score {risk['overall_score']}/100 ({risk['severity']}); Traffic ML {traffic_ml_pct:.1f}%, Flood ML {flood_ml_pct:.1f}%."

    return {
        "risk_assessment": risk,
        "risk_finding": {
            "agent": "risk_assessment_agent",
            "summary": summary,
            "reasoning_mode": llm_meta["reasoning_mode"],
            "findings": {
                "cross_domain_relationships": relationships,
                "cascading_risk_narrative": narrative,
                "uncertainty_note": uncertainty_note,
                "ml_scores": risk["ml_scores"],
                "llm_meta": llm_meta,
            },
        },
    }


def knowledge_agent(s: dict) -> dict:
    keywords = [s["scenario"].replace("_", " "), s["zone"]]
    if s["flood_finding"]["findings"]["indicator_level"] in {"high", "moderate"}:
        keywords.append("flood water level drainage sensor verification escalation")
    if s["traffic_finding"]["findings"]["congestion_level"] in {"high", "moderate"}:
        keywords.append("traffic congestion corridor diversion signal management")
    if s["environment_finding"]["findings"]["indicator_band"] in {"high", "moderate"}:
        keywords.append("air quality AQI health advisory")
    if s["infrastructure_finding"]["findings"]["maintenance_flag"] == "needs_inspection":
        keywords.append("storm drain culvert pump station infrastructure inspection")
    q = " ".join(keywords)
    docs = search_knowledge(q, 3)
    method = docs[0].get("retrieval_method", "rag") if docs else "rag"
    return {
        "knowledge_finding": {
            "agent": "knowledge_rag_agent",
            "summary": f"Retrieved {len(docs)} SOP chunk(s) via {method}.",
            "reasoning_mode": method,
            "findings": {"query": q, "retrieval_method": method, "evidence": docs},
        }
    }


def response_planner_agent(s: dict) -> dict:
    baseline_actions = [
        {"priority": "high", "action": "Verify source, timestamp, units and quality of readings.", "source_citation": "baseline_safety_protocol"},
        {"priority": "high", "action": "Have an authorised operations officer review official sources before any field action.", "source_citation": "human_in_the_loop_policy"},
    ]
    if s["flood_finding"]["findings"]["indicator_level"] == "high":
        baseline_actions.append({"priority": "high", "action": "Inspect drainage intakes, pump stations, and low-lying flood-prone locations.", "source_citation": "knowledge_base/sample_flood_sop.md"})
    if s["traffic_finding"]["findings"]["congestion_level"] == "high":
        baseline_actions.append({"priority": "medium", "action": "Review corridor signal timing and alternate routes using verified road-closure and flood-depth data.", "source_citation": "knowledge_base/sample_traffic_sop.md"})
    if s["environment_finding"]["findings"]["indicator_band"] == "high":
        baseline_actions.append({"priority": "medium", "action": "Prepare draft air-quality advisory for sensitive groups and cross-check regional AQI bulletins.", "source_citation": "knowledge_base/sample_environment_sop.md"})
    if s["infrastructure_finding"]["findings"]["maintenance_flag"] == "needs_inspection":
        baseline_actions.append({"priority": "high", "action": "Schedule public works inspection for blocked storm grates and backup drainage pumps.", "source_citation": "knowledge_base/sample_infrastructure_sop.md"})

    evidence_snippets = [
        {"source": d.get("source"), "section": d.get("section", ""), "excerpt": d.get("text", "")[:450]}
        for d in s["knowledge_finding"]["findings"].get("evidence", [])
    ]

    llm_out, llm_meta = call_llm_json(
        system_prompt=(
            "You are the Response Planning Agent in CITYPULSE AI. Using the risk findings and retrieved SOP evidence, "
            "produce prioritized human-reviewable operational recommendations. "
            "Return JSON with keys: 'actions' (list of objects with 'priority' in high|medium|low, 'action', 'rationale', 'source_citation') "
            "and 'coordination_summary' (1-2 sentences). "
            "Every action MUST be phrased as a recommendation requiring human approval; never claim automatic dispatch."
        ),
        payload={
            "zone": s["zone"],
            "severity": s["risk_assessment"]["severity"],
            "risk_summary": s["risk_finding"]["summary"],
            "cross_domain_relationships": s["risk_finding"]["findings"]["cross_domain_relationships"],
            "retrieved_sop_evidence": evidence_snippets,
            "baseline_actions": baseline_actions,
        },
        schema_cls=ResponsePlanLLMOutput,
        use_llm=s.get("use_llm"),
    )

    if llm_out is not None:
        actions = [item.model_dump() for item in llm_out.actions]
        summary = f"LLM synthesized {len(actions)} SOP-grounded recommendations for review. {llm_out.coordination_summary}"
    else:
        actions = baseline_actions
        summary = f"Prepared {len(actions)} recommendations for human review."

    plan = {
        "zone": s["zone"],
        "risk_severity": s["risk_assessment"]["severity"],
        "actions": actions,
        "requires_human_approval": True,
        "execution_status": "recommendation_only",
        "reasoning_mode": llm_meta["reasoning_mode"],
        "safety_note": "No real-world action is executed by this prototype. Authorised human review is required.",
    }
    return {
        "response_plan": plan,
        "response_finding": {
            "agent": "response_planning_agent",
            "summary": summary,
            "reasoning_mode": llm_meta["reasoning_mode"],
            "findings": {"action_count": len(actions), "llm_meta": llm_meta},
        },
    }


def communication_agent(s: dict) -> dict:
    fallback_summary = (
        f"CITYPULSE analysis for {s['zone']}: illustrative risk is {s['response_plan']['risk_severity']} "
        f"({s['risk_assessment']['overall_score']}/100); {len(s['response_plan']['actions'])} recommendations await human review."
    )

    llm_out, llm_meta = call_llm_json(
        system_prompt=(
            "You are the Communication & Reporting Agent in CITYPULSE AI. "
            "Draft concise operational reporting materials strictly labeled as draft_only for authorised review. "
            "Return JSON with keys: 'report_summary' (1 sentence), 'executive_briefing' (3-4 sentences summarizing ML scores, risks, and top recommendations), "
            "and 'citizen_advisory_draft' (1-2 sentence non-alarmist draft pending official approval)."
        ),
        payload={
            "zone": s["zone"],
            "scenario": s["scenario"],
            "risk_assessment": s["risk_assessment"],
            "top_actions": s["response_plan"]["actions"][:4],
            "sensor_source": s["readings"].get("source", "simulated_demo"),
        },
        schema_cls=CommunicationLLMOutput,
        use_llm=s.get("use_llm"),
    )

    if llm_out is not None:
        report_summary = llm_out.report_summary
        executive_briefing = llm_out.executive_briefing
        citizen_advisory = llm_out.citizen_advisory_draft
    else:
        report_summary = fallback_summary
        executive_briefing = (
            f"{fallback_summary} Traffic ML predicted {s['risk_assessment']['ml_scores']['traffic_congestion_pct']:.1f}% congestion; "
            f"Flood ML estimated {s['risk_assessment']['ml_scores']['flood_risk_pct']:.1f}% flood risk. "
            f"Readings source: {s['readings'].get('source', 'simulated_demo')}."
        )
        citizen_advisory = (
            f"[DRAFT PENDING APPROVAL] City operations is monitoring {s['scenario'].replace('_', ' ')} indicators in {s['zone']}. "
            "Please rely on official municipal channels for verified advisories."
        )

    return {
        "communication_finding": {
            "agent": "communication_reporting_agent",
            "summary": report_summary,
            "reasoning_mode": llm_meta["reasoning_mode"],
            "findings": {
                "notification_status": "draft_only",
                "report_summary": report_summary,
                "executive_briefing": executive_briefing,
                "citizen_advisory_draft": citizen_advisory,
                "llm_meta": llm_meta,
            },
        }
    }
