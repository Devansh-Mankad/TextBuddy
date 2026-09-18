import json
from pathlib import Path
from pipeline import process

root=Path(__file__).parent/"tests"/"samples"
rows=[]
for path in sorted(root.glob("sample_*.txt"), key=lambda p:int(p.stem.split("_")[1])):
    result=process(path.read_text(encoding="utf-8"))
    rows.append({
        "sample":path.name,
        **result["summary"]
    })
print(json.dumps(rows,indent=2))
