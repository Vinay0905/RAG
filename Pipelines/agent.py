import json
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector
from fastembed import TextEmbedding, SparseTextEmbedding
from langgraph.graph import StateGraph, END

import config
import prompts
from reranker import CrossEncoderReranker

class RAGAgentPipeline:
    def __init__(self):
        print("Compiling agent nodes...")
        self.groq_client = Groq(api_key=config.GROQ_API_KEY)
        self.qdrant_client = QdrantClient(config.QDRANT_HOST)
        self.dense_model = TextEmbedding(config.DENSE_MODEL_NAME)
        self.sparse_model = SparseTextEmbedding(config.SPARSE_MODEL_NAME)
        self.reranker = CrossEncoderReranker()
        self.app = self.build_graph()

    def rrf(self, results_lists: list) -> list:
        scores = {}
        for rank_list in results_lists:
            for idx, hit in enumerate(rank_list):
                payload = hit.payload
                c_id = payload["chunk_id"]
                score = 1.0 / (60 + idx + 1)
                
                if c_id not in scores:
                    scores[c_id] = {"payload": payload, "score": 0.0}
                scores[c_id]["score"] += score
        sorted_scores = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["payload"] for item in sorted_scores]

    def expand_query(self, state: dict) -> dict:
        query = state["current_query"]
        comp = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompts.QUERY_EXPANSION_PROMPT.format(query=query)}],
            model=config.EXPANSION_MODEL,
            temperature=0.2
        )
        resp = comp.choices[0].message.content.strip()
        queries = [q.strip() for q in resp.split("\n") if q.strip()]
        if query not in queries:
            queries.append(query)
        return {"expanded_queries": queries}

    def retrieve(self, state: dict) -> dict:
        queries = state["expanded_queries"]
        results = []
        for q in queries:
            q_dense = list(self.dense_model.embed([q]))[0].tolist()
            res = self.qdrant_client.search(
                collection_name=config.COLLECTION_NAME,
                query_vector=("dense-bge", q_dense),
                limit=10
            )
            results.append(res)
        return {"retrieved_chunks": self.rrf(results)}

    def rerank(self, state: dict) -> dict:
        top_chunks = self.reranker.rerank(state["original_query"], state["retrieved_chunks"])
        return {"reranked_chunks": top_chunks}

    def generate(self, state: dict) -> dict:
        query = state["original_query"]
        contexts = ""
        for idx, c in enumerate(state["reranked_chunks"]):
            contexts += f"\nSOURCE [{idx+1}]: {c['contract_name']} ({c['year']}) Section: {c['section']}\nContent: {c['chunk_text']}\n"
            
        comp = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompts.GENERATION_PROMPT.format(query=query, contexts=contexts)}],
            model=config.GENERATION_MODEL,
            temperature=0.0
        )
        return {"response": comp.choices[0].message.content.strip()}

    def grade_response(self, state: dict) -> dict:
        query = state["original_query"]
        resp = state["response"]
        context = "\n\n".join([c["chunk_text"] for c in state["reranked_chunks"]])
        
        if "INSUFFICIENT_CONTEXT" in resp:
            return {"grading_report": {"grounded": False, "relevant": False}}
            
        comp = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompts.GRADER_PROMPT.format(query=query, context=context, response=resp)}],
            model=config.GRADER_MODEL,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        report = json.loads(comp.choices[0].message.content.strip())
        return {"grading_report": report}

    def feedback_rewrite(self, state: dict) -> dict:
        query = state["current_query"]
        comp = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompts.QUERY_REWRITE_PROMPT.format(query=query)}],
            model=config.GRADER_MODEL,
            temperature=0.3
        )
        return {"current_query": comp.choices[0].message.content.strip(), "retry_count": state["retry_count"] + 1}

    def route_grading(self, state: dict) -> str:
        report = state["grading_report"]
        if report.get("grounded", False) and report.get("relevant", False):
            return "pass"
        if state["retry_count"] < state["max_retries"]:
            return "fail"
        return "max_reached"

    def build_graph(self):
        workflow = StateGraph(dict)
        workflow.add_node("expand_query", self.expand_query)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("rerank", self.rerank)
        workflow.add_node("generate", self.generate)
        workflow.add_node("grade_response", self.grade_response)
        workflow.add_node("feedback_rewrite", self.feedback_rewrite)
        
        workflow.set_entry_point("expand_query")
        workflow.add_edge("expand_query", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "grade_response")
        
        workflow.add_conditional_edges(
            "grade_response",
            self.route_grading,
            {"pass": END, "fail": "feedback_rewrite", "max_reached": END}
        )
        workflow.add_edge("feedback_rewrite", "expand_query")
        return workflow.compile()

    def run_agent(self, query: str):
        initial_state = {
            "original_query": query,
            "current_query": query,
            "expanded_queries": [],
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "response": "",
            "retry_count": 0,
            "max_retries": config.MAX_RETRIES,
            "grading_report": {}
        }
        return self.app.invoke(initial_state)