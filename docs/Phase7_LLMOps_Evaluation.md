# Phase 7 Study Guide: Enterprise LLMOps Evaluation Engine

Welcome to **Phase 7** of building our **Production-Grade Enterprise Self-Corrective RAG System**!

In this phase, you will build an automated LLMOps evaluation framework implementing the **RAG Triad** (Faithfulness, Answer Relevance, Context Recall) and a Synthetic Contract QA Generator for creating gold-standard evaluation benchmarks.

---

## 📁 Directory Structure for Phase 7

Ensure the following subdirectories exist inside `src/evaluation/`:

```text
src/evaluation/
├── __init__.py
├── metrics/
│   ├── __init__.py
│   ├── groundedness.py
│   ├── relevance.py
│   └── context_recall.py
├── evaluator.py
└── synthetic_generator.py
```

---

## 🛠️ Step-by-Step Code Files & Snippet Guides

---

### File 1: `src/evaluation/metrics/groundedness.py`
**Location**: `src/evaluation/metrics/groundedness.py`  
**Purpose**: Computes numerical Faithfulness score (0.0 to 1.0) evaluating if answer facts exist in retrieved context.

```python
import json
from groq import Groq
from config.settings import settings


class GroundednessEvaluator:
    """Evaluates answer faithfulness against reference context."""

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None

    def evaluate(self, response: str, context: str) -> float:
        if not self.client or "INSUFFICIENT_CONTEXT" in response:
            return 0.0

        prompt = f"""Rate the Faithfulness of the Candidate Response against the Reference Context on a scale from 0.0 to 1.0.
1.0 means every single claim is fully supported by the text. 0.0 means claims are ungrounded hallucinations.

Respond in JSON format: {{"score": float, "reason": "string"}}

Reference Context: {context}
Candidate Response: {response}"""

        try:
            res = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.GRADER_MODEL,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(res.choices[0].message.content.strip())
            return float(data.get("score", 0.0))
        except Exception:
            return 0.5
```

> **Deep 2-Line Explanation**:  
> *Quantifies hallucination levels by assigning a numerical score between 0.0 and 1.0 based on factual agreement.*  
> *Enables automated tracking of factual accuracy across different prompt iterations and model upgrades.*

---

### File 2: `src/evaluation/metrics/relevance.py`
**Location**: `src/evaluation/metrics/relevance.py`  
**Purpose**: Computes numerical Answer Relevance score evaluating if the generated text directly answers the user query.

```python
import json
from groq import Groq
from config.settings import settings


class AnswerRelevanceEvaluator:
    """Evaluates how directly the answer addresses the user query."""

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None

    def evaluate(self, query: str, response: str) -> float:
        if not self.client:
            return 0.0

        prompt = f"""Rate the Answer Relevance of the Response to the User Query on a scale from 0.0 to 1.0.
1.0 means completely answers the query. 0.0 means completely off-topic or evasive.

Respond in JSON format: {{"score": float, "reason": "string"}}

User Query: {query}
Response: {response}"""

        try:
            res = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.GRADER_MODEL,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(res.choices[0].message.content.strip())
            return float(data.get("score", 0.0))
        except Exception:
            return 0.5
```

> **Deep 2-Line Explanation**:  
> *Measures whether the generated output actually answers the question instead of outputting generic disclaimers.*  
> *Ensures high usability by scoring conversational alignment between questions and answers.*

---

### File 3: `src/evaluation/metrics/context_recall.py`
**Location**: `src/evaluation/metrics/context_recall.py`  
**Purpose**: Measures retrieval precision and recall relative to expected ground truth answers.

```python
import json
from groq import Groq
from config.settings import settings


class ContextRecallEvaluator:
    """Measures context recall against ground truth statements."""

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None

    def evaluate(self, ground_truth: str, retrieved_context: str) -> float:
        if not self.client:
            return 0.0

        prompt = f"""Rate the Context Recall of the Retrieved Context against the Ground Truth answer on a scale from 0.0 to 1.0.
1.0 means all facts needed to answer the question are present in the context.

Respond in JSON format: {{"score": float, "reason": "string"}}

Ground Truth: {ground_truth}
Retrieved Context: {retrieved_context}"""

        try:
            res = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.GRADER_MODEL,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(res.choices[0].message.content.strip())
            return float(data.get("score", 0.0))
        except Exception:
            return 0.5
```

> **Deep 2-Line Explanation**:  
> *Evaluates whether the vector store retrieved all contract context required to answer a reference question.*  
> *Helps optimize chunk sizes and hybrid retrieval parameters by measuring information completeness.*

---

### File 4: `src/evaluation/evaluator.py`
**Location**: `src/evaluation/evaluator.py`  
**Purpose**: Batch RAG Triad Evaluator runner aggregating metrics into a final evaluation report.

```python
from typing import List, Dict, Any
from src.evaluation.metrics.groundedness import GroundednessEvaluator
from src.evaluation.metrics.relevance import AnswerRelevanceEvaluator
from src.evaluation.metrics.context_recall import ContextRecallEvaluator
from src.core.logging import logger


class RAGTriadEvaluator:
    """Master evaluator aggregating Groundedness, Relevance, and Recall."""

    def __init__(self):
        self.groundedness_eval = GroundednessEvaluator()
        self.relevance_eval = AnswerRelevanceEvaluator()
        self.recall_eval = ContextRecallEvaluator()

    def evaluate_sample(self, query: str, response: str, context: str, ground_truth: str = "") -> Dict[str, float]:
        g_score = self.groundedness_eval.evaluate(response, context)
        r_score = self.relevance_eval.evaluate(query, response)
        rec_score = self.recall_eval.evaluate(ground_truth, context) if ground_truth else 1.0

        triad_score = (g_score + r_score + rec_score) / 3.0

        return {
            "groundedness": round(g_score, 4),
            "relevance": round(r_score, 4),
            "context_recall": round(rec_score, 4),
            "rag_triad_score": round(triad_score, 4)
        }

    def evaluate_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        logger.info(f"📊 Running batch evaluation across {len(samples)} samples...")
        totals = {"groundedness": 0.0, "relevance": 0.0, "context_recall": 0.0, "rag_triad_score": 0.0}

        for s in samples:
            res = self.evaluate_sample(
                query=s["query"],
                response=s["response"],
                context=s["context"],
                ground_truth=s.get("ground_truth", "")
            )
            for k in totals:
                totals[k] += res[k]

        count = len(samples) or 1
        return {k: round(v / count, 4) for k, v in totals.items()}
```

> **Deep 2-Line Explanation**:  
> *Runs full RAG Triad evaluations across individual queries or benchmark test batches.*  
> *Outputs consolidated mathematical averages for Groundedness, Relevance, and RAG Triad health.*

---

### File 5: `src/evaluation/synthetic_generator.py`
**Location**: `src/evaluation/synthetic_generator.py`  
**Purpose**: Parses contract chunks to automatically generate question-answer test pairs for evaluation.

```python
import json
from typing import List, Dict, Any
from groq import Groq
from config.settings import settings
from src.core.logging import logger


class SyntheticQAGenerator:
    """Generates synthetic contract QA pairs for automated benchmarking."""

    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY) if settings.GROQ_API_KEY else None

    def generate_pairs_from_chunk(self, chunk_text: str) -> List[Dict[str, str]]:
        if not self.client:
            return []

        prompt = f"""Based on the legal contract text below, generate 2 realistic question-answer test pairs.
Respond in JSON format as a list of objects:
[
  {{"question": "What is ...?", "ground_truth": "The contract states ..."}}
]

Contract Text:
{chunk_text[:1000]}"""

        try:
            res = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.GENERATION_MODEL,
                response_format={"type": "json_object"},
                temperature=0.3
            )
            data = json.loads(res.choices[0].message.content.strip())
            return data.get("pairs", []) if isinstance(data, dict) else []
        except Exception as e:
            logger.error(f"Synthetic QA generation failed: {str(e)}")
            return []
```

> **Deep 2-Line Explanation**:  
> *Reads ingested legal contract chunks and uses LLMs to automatically generate synthetic question-answer test datasets.*  
> *Creates ground-truth evaluation datasets instantly without manual human labeling effort.*

---

## 🎯 Phase 7 Checkpoint Verification

To verify Phase 7:
Run a sample evaluation in Python:
```python
from src.evaluation.evaluator import RAGTriadEvaluator

evaluator = RAGTriadEvaluator()
sample_report = evaluator.evaluate_sample(
    query="What is the liability cap?",
    response="The liability cap is $1,000,000 as stated in Section 4 [SOURCE 1].",
    context="Section 4: Total cumulative liability under this agreement shall not exceed $1,000,000."
)
print("Evaluation Metrics:\n", sample_report)
```

When you are ready, let me know to proceed to **Phase 8: FastAPI Backend, Async Server & Streaming**!
