from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
docs = {x["document_id"]: x for x in manifest}
assert len(docs) == 25
for item in manifest:
    data = (root / item["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == item["sha256"], item["path"]
ids = set()
count = 0
for path in sorted((root / "evaluation").glob("*.jsonl")):
    for line in path.read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        assert q["id"] not in ids
        ids.add(q["id"])
        assert q["split"] == path.stem
        assert q["required_claims"]
        for ref in q["evidence"]:
            text = (root / docs[ref["document_id"]]["path"]).read_text(encoding="utf-8")
            assert "## " + ref["section"] + ":" in text, ref
        count += 1
assert count == 30
print("OK: 25 documents, 30 unique questions; all evidence references and file hashes are valid.")
