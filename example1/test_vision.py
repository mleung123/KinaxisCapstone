# -*- coding: utf-8 -*-
# test_vision.py - Quick test for LM Studio vision API.
#
# Run from cmd/powershell:
#   python test_vision.py path\to\Newsboy_Model_with_Pricing.pdf

import base64, sys, json

# ---- CONFIG - edit these if needed ----
LM_URL  = "http://localhost:1234"
MODEL   = "qwen/qwen2.5-vl-7b-instruct"  # adjust to match LM Studio exactly
PAGE    = 0                               # 0-indexed
TIMEOUT = 600                             # seconds - vision encoding is slow
DPI     = 96                              # lower than 150 = faster, still readable
# ----------------------------------------

def render_page_b64(pdf_path, page_num, dpi):
    import fitz
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=dpi)
    return base64.b64encode(pix.tobytes("png")).decode("utf-8")

def test_vision(pdf_path):
    import requests
    print("Rendering page %d of %s at %d DPI..." % (PAGE+1, pdf_path, DPI))
    b64 = render_page_b64(pdf_path, PAGE, DPI)
    print("Image size: %dKB base64" % (len(b64)//1024))
    print("Timeout set to %ds" % TIMEOUT)

    payload = {
        "model": MODEL,
        "max_tokens": 500,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                {"type": "text", "text": "Describe what you see on this slide in 2-3 sentences."}
            ]
        }]
    }

    print("Sending to %s/v1/chat/completions ..." % LM_URL)
    print("(This will take 2-4 minutes on first run - do not cancel)")
    r = requests.post(LM_URL + "/v1/chat/completions", json=payload, timeout=TIMEOUT)
    print("\nHTTP %d" % r.status_code)
    print("--- RAW RESPONSE ---")
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text[:2000])

if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "Newsboy_Model_with_Pricing.pdf"
    test_vision(pdf)