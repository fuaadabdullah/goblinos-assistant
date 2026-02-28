from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from .config import settings
from .services.rag_service import RAGService


router = APIRouter(prefix="/rag", tags=["rag"])


def _resolve_collection_name(base_collection: str, content_type: str) -> str:
    base = (base_collection or "documents").strip() or "documents"
    ct = (content_type or "general").strip().lower() or "general"
    if ct in ("general", "default"):
        return base
    if "__" in base:
        return base
    return f"{base}__{ct}"


def _require_api_key(x_api_key: str | None) -> None:
    required = os.getenv("RAG_API_KEY", "").strip()
    if not required:
        return
    if x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid API key")


class DocumentItem(BaseModel):
    content: str = Field(..., min_length=1)
    id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AddDocumentsRequest(BaseModel):
    documents: List[DocumentItem]
    collection: str = "documents"
    content_type: str = "general"  # general|code|legal|scientific (routes to collection__content_type)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    top_k: int = Field(10, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None
    use_hybrid: bool = True
    use_reranking: bool = True
    expand_query: bool = True
    collection: str = "documents"
    content_type: str = "general"


class TopicsRequest(BaseModel):
    # Either provide texts (+ optional embeddings) or point at a Chroma collection.
    texts: Optional[List[str]] = None
    ids: Optional[List[str]] = None
    embeddings: Optional[List[List[float]]] = None

    collection: str = "documents"
    content_type: str = "general"
    max_items: int = Field(2000, ge=1, le=20000)

    algorithm: str = "auto"  # auto|hdbscan|kmeans
    n_clusters: int = Field(12, ge=2, le=200)
    min_cluster_size: int = Field(5, ge=2, le=500)


class DuplicatesRequest(BaseModel):
    texts: Optional[List[str]] = None
    ids: Optional[List[str]] = None
    embeddings: Optional[List[List[float]]] = None

    collection: str = "documents"
    content_type: str = "general"
    max_items: int = Field(2000, ge=1, le=20000)

    similarity_threshold: float = Field(0.92, ge=0.5, le=0.999)


def _embed_passages(
    rag_service: RAGService, texts: List[str], *, content_type: str = "general"
) -> List[List[float]]:
    model = getattr(rag_service, "get_embedder", None)
    model = model(content_type) if callable(model) else getattr(rag_service, "embedding_model", None)
    if not model:
        raise HTTPException(
            status_code=503,
            detail="Embedding model unavailable (install sentence-transformers or provide embeddings).",
        )

    if hasattr(model, "encode_passage"):
        vecs = model.encode_passage(texts)
    else:
        vecs = model.encode(texts)

    if hasattr(vecs, "tolist"):
        return vecs.tolist()

    return [list(v) for v in vecs]


def _load_collection_items(
    rag_service: RAGService, *, collection_name: str, max_items: int
) -> tuple[list[str], list[str], list[list[float]]]:
    if not rag_service.chroma_client:
        raise HTTPException(
            status_code=503,
            detail="ChromaDB not available; provide texts/embeddings instead.",
        )

    collection = rag_service.chroma_client.get_or_create_collection(name=collection_name)
    data = collection.get(
        limit=max_items,
        include=["documents", "embeddings"],
    )

    ids = [str(i) for i in (data.get("ids") or [])]
    texts = [str(t) for t in (data.get("documents") or [])]
    embeddings = data.get("embeddings") or []

    if not ids or not texts or not embeddings:
        return [], [], []

    return ids, texts, embeddings


@router.get("/health")
async def rag_health(x_api_key: str | None = Header(default=None)):
    _require_api_key(x_api_key)

    rag_service = RAGService(
        enable_enhanced=settings.enable_enhanced_rag,
        chroma_path=settings.rag_chroma_path,
    )

    return {
        "status": "healthy",
        "enhanced_rag_enabled": settings.enable_enhanced_rag,
        "chroma_available": bool(rag_service.chroma_client),
        "embedding_available": bool(getattr(rag_service, "embedding_model", None)),
    }


@router.post("/documents")
async def add_documents(
    req: AddDocumentsRequest, x_api_key: str | None = Header(default=None)
):
    _require_api_key(x_api_key)

    rag_service = RAGService(
        enable_enhanced=settings.enable_enhanced_rag,
        chroma_path=settings.rag_chroma_path,
    )

    docs = [
        {"content": d.content, "id": d.id, "metadata": d.metadata}
        for d in req.documents
    ]

    resolved_collection = _resolve_collection_name(req.collection, req.content_type)
    ok = await rag_service.add_documents(docs, resolved_collection, req.content_type)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to add documents")
    return {
        "status": "success",
        "count": len(docs),
        "collection": resolved_collection,
        "content_type": req.content_type,
    }


@router.post("/query")
async def rag_query(req: QueryRequest, x_api_key: str | None = Header(default=None)):
    _require_api_key(x_api_key)

    rag_service = RAGService(
        enable_enhanced=settings.enable_enhanced_rag,
        chroma_path=settings.rag_chroma_path,
    )

    resolved_collection = _resolve_collection_name(req.collection, req.content_type)

    if settings.enable_enhanced_rag:
        return await rag_service.enhanced_rag_pipeline(
            query=req.query,
            session_id=req.session_id,
            filters=req.filters,
            use_hybrid=req.use_hybrid,
            use_reranking=req.use_reranking,
            expand_query=req.expand_query,
            collection_name=resolved_collection,
            content_type=req.content_type,
            top_k=req.top_k,
        )

    return await rag_service.rag_pipeline(
        query=req.query,
        session_id=req.session_id,
        filters=req.filters,
        collection_name=resolved_collection,
        content_type=req.content_type,
        top_k=req.top_k,
    )


@router.post("/topics")
async def rag_topics(req: TopicsRequest, x_api_key: str | None = Header(default=None)):
    _require_api_key(x_api_key)

    rag_service = RAGService(
        enable_enhanced=settings.enable_enhanced_rag,
        chroma_path=settings.rag_chroma_path,
    )

    if req.texts is None:
        ids, texts, embeddings = _load_collection_items(
            rag_service,
            collection_name=_resolve_collection_name(req.collection, req.content_type),
            max_items=req.max_items,
        )
    else:
        texts = req.texts
        ids = req.ids or [str(i) for i in range(len(texts))]
        embeddings = req.embeddings or _embed_passages(
            rag_service, texts, content_type=req.content_type
        )

    if not ids:
        return {"status": "empty", "clusters": [], "noise_ids": [], "points": []}

    from .services.rag.semantic_analysis import discover_topics

    result = discover_topics(
        ids=ids,
        texts=texts,
        embeddings=embeddings,
        algorithm=req.algorithm,
        n_clusters=req.n_clusters,
        min_cluster_size=req.min_cluster_size,
    )

    return {
        "status": "success",
        "algorithm": result.algorithm,
        "clusters": [c.__dict__ for c in result.clusters],
        "noise_ids": result.noise_ids,
        "points": [p.__dict__ for p in result.points],
    }


@router.post("/duplicates")
async def rag_duplicates(
    req: DuplicatesRequest, x_api_key: str | None = Header(default=None)
):
    _require_api_key(x_api_key)

    rag_service = RAGService(
        enable_enhanced=settings.enable_enhanced_rag,
        chroma_path=settings.rag_chroma_path,
    )

    if req.texts is None:
        ids, _texts, embeddings = _load_collection_items(
            rag_service,
            collection_name=_resolve_collection_name(req.collection, req.content_type),
            max_items=req.max_items,
        )
    else:
        ids = req.ids or [str(i) for i in range(len(req.texts))]
        embeddings = req.embeddings or _embed_passages(
            rag_service, req.texts, content_type=req.content_type
        )

    from .services.rag.semantic_analysis import find_semantic_duplicates

    groups = find_semantic_duplicates(
        ids=ids,
        embeddings=embeddings,
        similarity_threshold=req.similarity_threshold,
    )

    return {
        "status": "success",
        "groups": groups,
        "group_count": len(groups),
    }
