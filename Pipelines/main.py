import argparse
import sys
from ingestion import IngestionPipeline
from agent import RAGAgentPipeline

def main():
    parser = argparse.ArgumentParser(description="Industrial CUAD Legal Contracts LangGraph RAG CLI Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    
    # Ingestion subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest CUAD contracts into Qdrant")
    ingest_parser.add_argument("--limit", type=int, default=100, help="Number of contracts to process and index")
    
    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Run the RAG query agent")
    query_parser.add_argument("text", type=str, help="Search query question")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        pipeline = IngestionPipeline()
        pipeline.run(limit=args.limit)
    elif args.command == "query":
        agent = RAGAgentPipeline()
        res = agent.run_agent(args.text)
        print("\n================ ANSWER ================")
        print(res["response"])
        print("========================================")
        print(f"Correction retries: {res['retry_count']}")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()