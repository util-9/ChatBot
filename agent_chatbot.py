import os
import faiss
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_openai import ChatOpenAI

load_dotenv()

class ControllableRAGAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("GROQ_API_KEY"),
            openai_api_base="https://api.groq.com/openai/v1",
            model="llama3-70b-8192"  # a Groq-supported model
        )
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        self.documents = []
        self.embeddings = None
        self.faiss_index = None
        self.bm25 = None

    def load_documents(self, directory):
        documents = []
        # Replace with your PDF/text loader for real use!
        domain_texts = [
            "Gasoline fuel injection improves combustion and efficiency...",
            "Droplet formation, cone angle of spray, and atomization are key...",
            "Basic fluid mechanics: viscosity affects spray breakup and penetration..."
        ]
        for text in domain_texts:
            chunks = self.text_splitter.split_text(text)
            for chunk in chunks:
                documents.append(Document(page_content=chunk))
        self.documents = documents
        self._build_indices()

    def _build_indices(self):
        texts = [doc.page_content for doc in self.documents]
        self.embeddings = self.embedding_model.encode(texts)
        self.faiss_index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.faiss_index.add(self.embeddings.astype('float32'))
        tokenized_docs = [doc.page_content.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def _expand_query(self, query: str) -> List[str]:
        prompt = (
            "Generate 3 technical paraphrases for this fuel injection engineering question:\n"
            f"{query}\n\nJust the questions, no commentary."
        )
        try:
            response = self.llm.invoke(prompt)
            variations = response.content.strip().split('\n')
            return [query] + [v.strip() for v in variations if v.strip()]
        except Exception:
            return [query]

    def _hybrid_retrieve(self, query: str, top_k=10) -> List[Dict]:
        expanded = self._expand_query(query)
        all_candidates = set()
        for exp_query in expanded:
            query_emb = self.embedding_model.encode([exp_query])
            dense_scores, dense_indices = self.faiss_index.search(query_emb.astype('float32'), top_k)
            tokenized_query = exp_query.split()
            sparse_scores = self.bm25.get_scores(tokenized_query)
            sparse_indices = np.argsort(sparse_scores)[::-1][:top_k]
            for i, idx in enumerate(dense_indices[0]):
                all_candidates.add((idx, dense_scores[0][i], 'dense'))
            for i, idx in enumerate(sparse_indices):
                all_candidates.add((idx, sparse_scores[idx], 'sparse'))
        candidates = []
        for idx, score, method in all_candidates:
            if idx < len(self.documents):
                candidates.append({
                    'document': self.documents[idx].page_content,
                    'score': float(score),
                    'method': method,
                    'index': idx
                })
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]

    def _planner_decompose(self, query: str) -> List[str]:
        prompt = (
            f"Break down the following technical question into stepwise sub-tasks for a fuel injection engineering RAG QA agent. "
            f"{query}\n"
            f"List the steps, one per line."
        )
        result = self.llm.invoke(prompt)
        return [s.strip() for s in result.content.splitlines() if s.strip()]

    def _multi_step_reason(self, main_query: str) -> str:
        steps = self._planner_decompose(main_query)
        context_accum = []

        for i, sub_q in enumerate(steps):
            retrieved = self._hybrid_retrieve(sub_q, top_k=3)
            context_text = "\n\n".join([d['document'] for d in retrieved])
            step_prompt = (
                f"As a mechanical engineering assistant, use ONLY the context below to answer the sub-task: {sub_q}\n"
                f"Context:\n{context_text}\n\nAnswer concisely and technically."
            )
            try:
                result = self.llm.invoke(step_prompt)
                answer = result.content.strip()
            except Exception as e:
                answer = f"[Agent error: {e}]"
            context_accum.append(f"Step {i+1}: {sub_q}\nA: {answer}")

        full_reasoning = "\n\n".join(context_accum)
        final_prompt = (
            f"For the original question '{main_query}', summarize and combine all the step answers below into a technical, fluent answer, "
            f"grounded strictly in provided info:\n\n{full_reasoning}"
        )
        try:
            final_result = self.llm.invoke(final_prompt)
            return final_result.content.strip()
        except Exception as e:
            return f"[Final Agent error: {e}]"

    def query(self, user_query: str) -> Dict[str, Any]:
        answer = self._multi_step_reason(user_query)
        return {
            'answer': answer,
            'plan': self._planner_decompose(user_query)
        }
