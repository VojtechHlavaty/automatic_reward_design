"""Retrieval database of earlier curricula, their outcomes and their feedback.

Records are stored as JSON lines in ``records.jsonl`` and their embeddings in ``embeddings.npy`` (unit vectors,
one row per record). Embeddings come from a local sentence-embedding model (BGE-small by default), so no API key
is needed. Search is cosine similarity over all records.
"""

import json
import os
import threading
import uuid

import numpy as np


class Embedder:
    """Sentence embeddings with a Hugging Face encoder (CLS pooling, unit length), on the CPU."""

    def __init__(self, model_name):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()

    def embed(self, texts):
        import torch

        with torch.no_grad():
            batch = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            vectors = self.model(**batch).last_hidden_state[:, 0]
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
        return vectors.numpy().astype(np.float32)


class VectorDB:
    def __init__(self, path, embedder):
        self.path = path
        self.embedder = embedder
        self.lock = threading.Lock()
        os.makedirs(path, exist_ok=True)
        self.records_file = os.path.join(path, "records.jsonl")
        self.embeddings_file = os.path.join(path, "embeddings.npy")
        self.records = []
        if os.path.exists(self.records_file):
            with open(self.records_file) as f:
                self.records = [json.loads(line) for line in f if line.strip()]
        self.embeddings = np.load(self.embeddings_file) if os.path.exists(self.embeddings_file) else None
        if self.embeddings is not None and len(self.embeddings) != len(self.records):
            # out of sync (e.g. interrupted write): recompute all embeddings
            self.embeddings = self.embedder.embed([r["text"] for r in self.records]) if self.records else None
            self._save()

    def __len__(self):
        return len(self.records)

    def _save(self):
        tmp = self.records_file + ".tmp"
        with open(tmp, "w") as f:
            for record in self.records:
                f.write(json.dumps(record) + "\n")
        os.replace(tmp, self.records_file)
        if self.embeddings is not None:
            np.save(self.embeddings_file + ".tmp.npy", self.embeddings)
            os.replace(self.embeddings_file + ".tmp.npy", self.embeddings_file)

    def add(self, record, text):
        """Add a record; ``text`` is the content that is embedded for retrieval. Returns the record id."""
        with self.lock:
            record = dict(record, id=record.get("id") or uuid.uuid4().hex, text=text)
            vector = self.embedder.embed([text])
            self.records.append(record)
            self.embeddings = vector if self.embeddings is None else np.concatenate([self.embeddings, vector])
            self._save()
            return record["id"]

    def has(self, record_id):
        return any(r["id"] == record_id for r in self.records)

    def update(self, record_id, **fields):
        """Update fields of a record (e.g. human feedback added later) and re-embed it."""
        with self.lock:
            for k, record in enumerate(self.records):
                if record["id"] == record_id:
                    record.update(fields)
                    if "text" in fields:
                        self.embeddings[k] = self.embedder.embed([record["text"]])[0]
                    self._save()
                    return True
            return False

    def search(self, query, k):
        """The ``k`` most similar records as (cosine similarity, record), best first."""
        if not self.records:
            return []
        vector = self.embedder.embed([query])[0]
        scores = self.embeddings @ vector
        order = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.records[i]) for i in order]
