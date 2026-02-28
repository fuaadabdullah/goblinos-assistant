from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import re
from sqlalchemy.orm import Session
from .database import get_db
from .models import SearchCollection, SearchDocument

router = APIRouter(prefix="/search", tags=["search"])


class SearchQuery(BaseModel):
    query: str
    collection_name: str = "documents"
    n_results: int = 10


class SearchResult(BaseModel):
    id: str
    document: str
    metadata: Optional[Dict[str, Any]] = None
    score: Optional[float] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total_results: int


def simple_text_search(
    query: str, documents: List[Dict], n_results: int = 10
) -> List[Dict]:
    """Simple text-based search implementation"""
    query_lower = query.lower()
    scored_docs = []

    for doc in documents:
        text = doc.get("document", "").lower()
        score = 0

        # Simple scoring based on word matches
        query_words = query_lower.split()
        for word in query_words:
            if word in text:
                score += 1

        # Boost score for exact phrase matches
        if query_lower in text:
            score += 10

        if score > 0:
            scored_docs.append(
                {
                    "id": doc.get("id", f"doc_{len(scored_docs)}"),
                    "document": doc.get("document", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": score,
                }
            )

    # Sort by score and return top results
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return scored_docs[:n_results]


@router.post("/query", response_model=SearchResponse)
async def search_documents(search_query: SearchQuery, db: Session = Depends(get_db)):
    """Search documents using simple text search"""
    try:
        collection = (
            db.query(SearchCollection)
            .filter(SearchCollection.name == search_query.collection_name)
            .first()
        )
        if not collection:
            return SearchResponse(results=[], total_results=0)

        # Get documents from collection
        documents = (
            db.query(SearchDocument)
            .filter(SearchDocument.collection_id == collection.id)
            .all()
        )

        if not documents:
            return SearchResponse(results=[], total_results=0)

        # Convert to dict format for search function
        docs_dict = []
        for doc in documents:
            docs_dict.append(
                {
                    "id": doc.document_id,
                    "document": doc.document,
                    "metadata": doc.metadata,
                }
            )

        # Perform search
        results = simple_text_search(
            search_query.query, docs_dict, search_query.n_results
        )

        # Format results
        search_results = []
        for result in results:
            search_results.append(
                SearchResult(
                    id=result["id"],
                    document=result["document"],
                    metadata=result["metadata"],
                    score=result["score"],
                )
            )

        return SearchResponse(results=search_results, total_results=len(search_results))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/collections")
async def list_collections(db: Session = Depends(get_db)):
    """List all available collections"""
    try:
        collections = db.query(SearchCollection).all()
        return {"collections": [c.name for c in collections]}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list collections: {str(e)}"
        )


@router.post("/collections/{collection_name}/add")
async def add_document(
    collection_name: str,
    document: str,
    metadata: Optional[Dict[str, Any]] = None,
    id: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Add a document to a collection"""
    try:
        # Get or create collection
        collection = (
            db.query(SearchCollection)
            .filter(SearchCollection.name == collection_name)
            .first()
        )
        if not collection:
            collection = SearchCollection(name=collection_name)
            db.add(collection)
            db.commit()
            db.refresh(collection)

        # Generate document ID if not provided
        doc_id = (
            id
            or f"doc_{db.query(SearchDocument).filter(SearchDocument.collection_id == collection.id).count()}"
        )

        # Create document
        search_doc = SearchDocument(
            document_id=doc_id,
            document=document,
            metadata=metadata or {},
            collection_id=collection.id,
        )
        db.add(search_doc)
        db.commit()
        db.refresh(search_doc)

        return {"status": "success", "document_id": doc_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add document: {str(e)}")


@router.get("/collections/{collection_name}/documents")
async def get_collection_documents(collection_name: str, db: Session = Depends(get_db)):
    """Get all documents in a collection"""
    try:
        collection = (
            db.query(SearchCollection)
            .filter(SearchCollection.name == collection_name)
            .first()
        )
        if not collection:
            return {"documents": []}

        documents = (
            db.query(SearchDocument)
            .filter(SearchDocument.collection_id == collection.id)
            .all()
        )

        # Convert to dict format for API compatibility
        docs_list = []
        for doc in documents:
            docs_list.append(
                {
                    "id": doc.document_id,
                    "document": doc.document,
                    "metadata": doc.metadata,
                }
            )

        return {"documents": docs_list}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get documents: {str(e)}"
        )


@router.get("/suggest", tags=["search"])
async def search_suggest(q: str = Query("")):
    """Get search suggestions"""
    if not q:
        return {"suggestions": []}

    # Mock suggestions based on common queries
    suggestions = [
        f"{q} tutorial",
        f"{q} best practices",
        f"{q} examples",
        f"how to {q}",
        f"{q} documentation",
    ]

    return {"suggestions": suggestions}
