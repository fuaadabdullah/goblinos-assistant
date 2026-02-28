# apps/goblin-assistant-root/backend/models/search.py
"""
Database models for search collections and documents.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    JSON,
    ForeignKey,
)
from sqlalchemy.orm import relationship
from database import Base


class SearchCollection(Base):
    __tablename__ = "search_collections"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)

    documents = relationship(
        "SearchDocument", back_populates="collection_obj", cascade="all, delete-orphan"
    )


class SearchDocument(Base):
    __tablename__ = "search_documents"
    id = Column(
        Integer, primary_key=True, index=True
    )  # Primary key for SearchDocument itself
    document_id = Column(
        String(50), unique=True, nullable=False
    )  # Unique ID for the document within a collection
    collection_id = Column(Integer, ForeignKey("search_collections.id"), nullable=False)
    document = Column(Text, nullable=False)
    document_metadata = Column(
        JSON
    )  # Renamed from document_metadata to avoid SQLAlchemy reserved keyword conflict

    collection_obj = relationship("SearchCollection", back_populates="documents")

    def to_dict(self):
        return {
            "document_id": self.document_id,
            "document": self.document,
            "metadata": self.document_metadata or {},
            "collection": self.collection_obj.name if self.collection_obj else None,
        }
