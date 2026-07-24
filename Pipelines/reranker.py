from sentence_transformers import CrossEncoder
import config


class CrossEncoderReranker:
    def __init__(self):
        print(f"Loading reranker model '{config.RERANK_MODEL_NAME}'...")
        self.model = CrossEncoder(config.RERANK_MODEL_NAME)
        
    def rerank(self, query: str, candidates: list, top_k: int = 5) -> list:
        if not candidates:
            return []

        pairs = [[query, item["chunk_text"]] for item in candidates]
        scores = self.model.predict(pairs)

        scored_candidates = []
        for idx, score in enumerate(scores):
            item = candidates[idx]
            item["rerank_score"] = float(score)
            scored_candidates.append(item)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_k]



# if __name__=="__main__":
#     Reranker=CrossEncoderReranker()
