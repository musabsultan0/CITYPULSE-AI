from backend.app.core.config import settings
from backend.app.rag import (
    FAISS_AVAILABLE,
    KB_DIR,
    VectorRAGStore,
    get_rag_status,
    rebuild_knowledge_index,
    search_knowledge,
)


def test_faiss_index_builds_and_persists():
    assert FAISS_AVAILABLE is True
    status = rebuild_knowledge_index()
    assert status["active_mode"] == "faiss"
    assert status["faiss_available"] is True
    assert status["indexed_documents"] >= 4
    assert status["indexed_chunks"] >= 4
    assert status["index_persisted"] is True


def test_faiss_semantic_retrieval_domain_ranking():
    traffic_hits = search_knowledge("arterial traffic corridor diversion and adaptive signals", top_k=3)
    assert len(traffic_hits) >= 1
    assert "sample_traffic_sop" in traffic_hits[0]["source"]
    assert traffic_hits[0]["retrieval_method"] == "faiss_embeddings"

    aqi_hits = search_knowledge("high AQI PM2.5 air quality health advisory for schools", top_k=3)
    assert len(aqi_hits) >= 1
    assert "sample_environment_sop" in aqi_hits[0]["source"]

    infra_hits = search_knowledge("blocked storm grates culvert bottleneck pump station", top_k=3)
    assert len(infra_hits) >= 1
    assert "sample_infrastructure_sop" in infra_hits[0]["source"]

    flood_hits = search_knowledge("water level sensor calibration flood incident escalation", top_k=3)
    assert len(flood_hits) >= 1
    assert "sample_flood_sop" in flood_hits[0]["source"]


def test_faiss_index_reloads_from_disk_and_reindexes_new_doc():
    # Fresh store instance should load directly from persisted FAISS index
    fresh_store = VectorRAGStore()
    status = fresh_store.ensure_indexed()
    assert status["active_mode"] == "faiss"
    assert status["indexed_chunks"] >= 4

    # Add temporary SOP doc and verify automatic re-indexing retrieves it
    tmp_doc = KB_DIR / "_test_heatwave_sop.md"
    try:
        tmp_doc.write_text(
            "# Sample Heatwave Cooling Center Protocol\n"
            "Open municipal hydration stations and cooling shelters when heat index exceeds 44C.\n",
            encoding="utf-8",
        )
        rebuild_knowledge_index()
        hits = search_knowledge("hydration stations cooling shelters extreme heatwave", top_k=2)
        assert any("_test_heatwave_sop.md" in h["source"] for h in hits)
    finally:
        if tmp_doc.exists():
            tmp_doc.unlink()
        rebuild_knowledge_index()


def test_keyword_mode_still_supported(monkeypatch):
    monkeypatch.setattr(settings, "rag_mode", "keyword")
    status = get_rag_status()
    assert status["active_mode"] == "keyword"
    hits = search_knowledge("flood sensor calibration escalation", top_k=2)
    assert len(hits) >= 1
    assert hits[0]["retrieval_method"] == "keyword"
