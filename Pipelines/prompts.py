QUERY_EXPANSION_PROMPT = """You are a legal research search assistant.
Generate exactly 3 variations of the following search query to help retrieve relevant clauses (such as termination, indemnity, limitation of liability) from a legal contract vector database.
Generate ONLY the 3 queries, one per line. Do not number them or add any introductory text.

Original Query: {query}"""

GENERATION_PROMPT = """You are an expert corporate lawyer. Answer the query based ONLY on the provided contract sources below.
Use inline citations when stating facts (e.g. "Fact [SOURCE 1]").
If the context does not contain enough information to answer the query, reply with: 'INSUFFICIENT_CONTEXT'

Query: {query}

Sources:
{contexts}
"""

GRADER_PROMPT = """Analyze if the Candidate Response is fully grounded and supported by the Reference Context (legal contracts).
Also check if it directly answers the User Query.
Respond in exactly this JSON format:
{{
  "grounded": true/false,
  "relevant": true/false,
  "explanation": "Short explanation"
}}

User Query: {query}
Reference Context: {context}
Candidate Response: {response}"""

QUERY_REWRITE_PROMPT = """The previous search query '{query}' did not retrieve enough contract clause information to answer the legal topic.
Rewrite this query to be broader, using legal synonyms, contract terms, or different search angles to retrieve relevant clauses.
Output ONLY the rewritten search query. No extra conversational text.

Query: {query}"""