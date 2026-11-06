"""Export PDF sections as JSONL for inspection and subsequent retrieval."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .pdf_loader import load_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--include-archived", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/sections.jsonl"))
    args = parser.parse_args()
    sections = load_corpus(args.root, include_archived=args.include_archived)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for section in sections:
            row = asdict(section) | {"citation_id": section.citation_id}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    count = len({section.document_id for section in sections})
    print(f"Loaded {count} documents and {len(sections)} sections -> {args.output}")


if __name__ == "__main__":
    main()
