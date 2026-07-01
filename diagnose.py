"""Quick diagnostic: shows what's happening at each pipeline stage."""
import json
import sys
import requests

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"
SAMPLE = "c:/Users/epicc/Desktop/OCR Module/samples/sample_receipt.jpg"

# ── 1. Ollama reachable? ────────────────────────────────────────────────────
print("=== 1. Ollama connectivity ===")
try:
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"  OK — models available: {models}")
    if not any(OLLAMA_MODEL in m for m in models):
        print(f"  WARNING: '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
except Exception as e:
    print(f"  FAIL — Ollama not reachable: {e}")
    print("  Make sure Ollama is running: ollama serve")
    sys.exit(1)

# ── 2. OCR stage — what text does PaddleOCR produce? ───────────────────────
print("\n=== 2. OCR text from sample ===")
try:
    import cv2
    from src.ocr_service.pipeline.config import Settings
    from src.ocr_service.pipeline.ocr_engine import run_ocr
    from src.ocr_service.pipeline.preprocessing import preprocess

    settings = Settings(ollama_enabled=False)
    image = cv2.imread(SAMPLE)
    if image is None:
        print(f"  FAIL — could not read image: {SAMPLE}")
        sys.exit(1)

    processed, applied = preprocess(image, settings)
    regions = run_ocr(processed, settings)
    print(f"  Regions detected: {len(regions)}")
    print("  Text per region:")
    for r in regions:
        print(f"    [{r.confidence:.2f}] {r.text!r}")
except ImportError as e:
    print(f"  SKIP — OCR deps not installed ({e})")
    print("  Install with: pip install -e '.[ocr]'")
    regions = []
except Exception as e:
    print(f"  FAIL — {e}")
    regions = []

# ── 3. LLM stage — what does Ollama return? ────────────────────────────────
print("\n=== 3. LLM extraction ===")

# Use OCR regions if available, otherwise use a hardcoded sample text
if regions:
    lines = []
    from src.ocr_service.pipeline.extraction import _group_rows
    rows = _group_rows(regions)
    for row in rows:
        line = " ".join(r.text for r in row).strip()
        if line:
            lines.append(line)
    text = "\n".join(lines)
else:
    text = "Invoice No: 1042\nDate: 2024-03-15\nVendor: ACME Corp\nTotal: 150.00 EUR"
    print("  (using fallback sample text since OCR regions unavailable)")

print(f"  Text sent to LLM ({len(text)} chars):\n---\n{text}\n---")

prompt = f"""\
You are digitizing a scanned financial document. Extract every labeled field.
Return ONLY a JSON object. Keys = field names, values = strings.

Document text:
{text}"""

payload = {
    "model": OLLAMA_MODEL,
    "prompt": prompt,
    "format": "json",
    "stream": False,
    "options": {"temperature": 0.0},
}

try:
    r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
    r.raise_for_status()
    raw = r.json().get("response", "")
    print(f"  Raw LLM response:\n{raw}")
    parsed = json.loads(raw)
    print(f"  Parsed fields ({len(parsed)}):")
    for k, v in parsed.items():
        print(f"    {k!r}: {v!r}")
except Exception as e:
    print(f"  FAIL — {e}")
