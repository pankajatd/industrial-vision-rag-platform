import os
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocumentChunker:
    """Splits markdown manuals into manageable chunks for retrieval."""
    
    @staticmethod
    def chunk_text(text: str, source: str) -> List[Dict[str, Any]]:
        # Simple split by double newline (paragraphs/sections)
        paragraphs = text.split("\n\n")
        chunks = []
        for p in paragraphs:
            p = p.strip()
            if len(p) > 20: # Ignore very short artifacts
                chunks.append({
                    "content": p,
                    "metadata": {"source": source}
                })
        return chunks

class BM25Index:
    """A lightweight local vector store using TF-IDF and Cosine Similarity."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        self.chunks = []
        self.tfidf_matrix = None
        self.is_fitted = False

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Adds documents to the index and fits the TF-IDF matrix."""
        self.chunks.extend(chunks)
        corpus = [chunk["content"] for chunk in self.chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_fitted = True

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves the top_k most relevant chunks for a given query."""
        if not self.is_fitted or len(self.chunks) == 0:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top K indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05: # Minimum threshold
                result = self.chunks[idx].copy()
                result["score"] = float(similarities[idx])
                results.append(result)
                
        return results

class VectorStore:
    """Wrapper that combines chunking and indexing."""
    
    def __init__(self):
        self.index = BM25Index()

    def ingest_files(self, filepaths: List[str]):
        """Reads files from disk, chunks them, and adds to index."""
        all_chunks = []
        for path in filepaths:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            filename = os.path.basename(path)
            chunks = DocumentChunker.chunk_text(content, source=filename)
            all_chunks.extend(chunks)
        
        self.index.add_chunks(all_chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.index.search(query, top_k=top_k)
