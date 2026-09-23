import argparse
import json
from pathlib import Path
from .config import Settings
from .service import RAGService


def main():
    parser = argparse.ArgumentParser(description="Multimodal PDF RAG")
    parser.add_argument("--tenant", default="local")
    parser.add_argument("--session", default="default")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest")
    ingest.add_argument("pdf", type=Path)
    ingest.add_argument("--name", help="Stable logical PDF name used for versioning")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--document")
    ask.add_argument("--kind", choices=["text", "table", "image"])
    ask.add_argument("--stream", action="store_true")
    sub.add_parser("documents")
    remember = sub.add_parser("remember")
    remember.add_argument("text")
    sub.add_parser("memory")
    sub.add_parser("clear-memory")
    args = parser.parse_args()
    try:
        service = RAGService(Settings.load(), args.tenant)
        if args.command == "ingest":
            result = service.ingest(args.pdf, args.name)
        elif args.command == "ask":
            if args.stream:
                for event in service.stream(args.question, args.session, args.document, args.kind):
                    print(json.dumps(event, ensure_ascii=False), flush=True)
                return
            result = service.ask(args.question, args.session, args.document, args.kind)
        elif args.command == "documents":
            result = service.catalog.documents()
        elif args.command == "remember":
            result = {"id": service.remember(args.session, args.text)}
        elif args.command == "memory":
            result = {"recent": service.catalog.history(args.session),
                      "long_term": service.catalog.memories(args.session)}
        else:
            service.catalog.clear_memory(args.session)
            result = {"status": "cleared"}
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
