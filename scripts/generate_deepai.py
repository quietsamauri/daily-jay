# scripts/generate_deepai.py
# Replicate + InstantID with: auto version fetch, per-day run suffixes, manifest, and safe fallback.

import os, datetime, json, pathlib, urllib.request, time, requests, base64, mimetypes, sys

PROMPTS_FILE = "prompts.txt"
OUT_DIR = "daily"
ASSETS_DIR = "assets"
TZ_OFFSET_HOURS = -4  # Rough ET offset for filenames

REPLICATE_MODEL = "zsxkib/instant-id"  # We'll fetch its latest version dynamically

def log(msg): print(f"[generator] {msg}", flush=True)

def choose_prompt_and_ref():
    if not os.path.exists(PROMPTS_FILE):
        raise RuntimeError(f"Missing {PROMPTS_FILE}")
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = [p.strip() for p in f if p.strip()]
    if not prompts:
        raise RuntimeError("prompts.txt is empty.")

    today = datetime.datetime.utcnow() + datetime.timedelta(hours=TZ_OFFSET_HOURS)
    daynum = int(today.strftime("%j"))
    prompt = prompts[daynum % len(prompts)]
    date_str = today.strftime("%Y-%m-%d")

    if not os.path.isdir(ASSETS_DIR):
        raise RuntimeError(f"Missing {ASSETS_DIR}/ directory")
    refs = sorted([f for f in os.listdir(ASSETS_DIR)
                   if f.lower().startswith("jay") and f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not refs:
        raise RuntimeError("No reference images in assets/ (expected jay1.jpg, jay2.jpg, jay3.jpg)")
    ref = os.path.join(ASSETS_DIR, refs[daynum % len(refs)])
    return prompt, ref, date_str

def replicate_latest_version(api_token):
    """Fetch the latest version id for the InstantID model."""
    url = f"https://api.replicate.com/v1/models/{REPLICATE_MODEL}/versions"
    headers = {"Authorization": f"Token {api_token}"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    versions = data.get("results") or []
    if not versions:
        raise RuntimeError(f"No versions returned for model {REPLICATE_MODEL}")
    return versions[0]["id"]  # newest first

def replicate_instantid(prompt, api_token, ref_path, version_id, width=896, height=1152, steps=28, guidance=6.5):
    """Call Replicate InstantID and return a direct image URL."""
    with open(ref_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    mime = mimetypes.guess_type(ref_path)[0] or "image/jpeg"
    data_url = f"data:{mime};base64,{b64}"

    endpoint = "https://api.replicate.com/v1/predictions"
    headers = {"Authorization": f"Token {api_token}", "Content-Type": "application/json"}
    payload = {
        "version": version_id,
        "input": {
            "prompt": prompt + ", ultra-detailed, photorealistic, natural skin tones, high dynamic range",
            "negative_prompt": "blurry, deformed, extra limbs, double face, watermark, text, low quality",
            "image": data_url,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "ip_adapter_scale": 0.85,
            "controlnet_conditioning_scale": 0.85,
            "seed": int(datetime.datetime.utcnow().timestamp()) % 2147483647
        }
    }

    r = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    poll_url = r.json()["urls"]["get"]

    for _ in range(120):  # up to ~4 minutes
        pr = requests.get(poll_url, headers=headers, timeout=30)
        pr.raise_for_status()
        p = pr.json()
        status = p.get("status")
        if status == "succeeded":
            out = p.get("output")
            if isinstance(out, list) and out:
                return out[-1]
            if isinstance(out, str):
                return out
            raise RuntimeError(f"Unexpected output format: {out}")
        if status in ("failed", "canceled"):
            raise RuntimeError(f"Replicate failed: {p.get('error') or status}")
        time.sleep(2)

    raise RuntimeError("Replicate timed out")

def download(url, dest, attempts=5):
    for i in range(attempts):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as e:
            log(f"download attempt {i+1} failed: {e}")
            time.sleep(1 + i)
    raise RuntimeError("Failed to download image.")

def load_manifest(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_manifest(path, items):
    items_sorted = sorted(items, key=lambda x: (x["date"], x["src"]), reverse=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items_sorted, f, indent=2)

def pick_asset_fallback():
    if not os.path.isdir(ASSETS_DIR):
        return None
    refs = [f for f in os.listdir(ASSETS_DIR)
            if f.lower().startswith("jay") and f.lower().endswith((".jpg",".jpeg",".png"))]
    if not refs:
        return None
    daynum = int(datetime.datetime.utcnow().strftime("%j"))
    return os.path.join(ASSETS_DIR, sorted(refs)[daynum % len(refs)])

def main():
    # Prep
    prompt, ref_image, date_str = choose_prompt_and_ref()
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Per-day run suffix
    base_name = date_str
    existing = [f for f in os.listdir(OUT_DIR) if f.startswith(base_name + "-") and f.endswith(".jpg")]
    run_num = len(existing) + 1
    file_name = f"{base_name}-{run_num}.jpg"

    today_path = f"{OUT_DIR}/{file_name}"
    latest_path = f"{OUT_DIR}/latest.jpg"

    meta = {"date": date_str, "prompt": prompt, "ref": os.path.basename(ref_image)}

    # Try Replicate (fallback to asset if anything goes wrong)
    used_fallback = False
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise RuntimeError("REPLICATE_API_TOKEN not set (add it as a repo Secret).")

        log("Fetching latest Replicate model version…")
        version_id = replicate_latest_version(api_token)
        log(f"Using {REPLICATE_MODEL}@{version_id}")
        img_url = replicate_instantid(prompt, api_token, ref_image, version_id)
        log(f"Image URL: {img_url}")
        download(img_url, today_path)
        meta["source"] = "replicate-instantid"
    except Exception as e:
        log(f"Generation failed, using fallback asset: {e}")
        fb = pick_asset_fallback()
        if not fb:
            print(f"[generator] FATAL: {e}")
            sys.exit(1)
        with open(fb, "rb") as src, open(today_path, "wb") as dst:
            dst.write(src.read())
        meta.update({"source": "fallback-asset", "error": str(e)})
        used_fallback = True

    # Update latest.jpg
    with open(today_path, "rb") as src, open(latest_path, "wb") as dst:
        dst.write(src.read())

    # Write meta + manifest
    with open(f"{OUT_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    manifest_path = f"{OUT_DIR}/manifest.json"
    manifest = load_manifest(manifest_path)
    manifest = [m for m in manifest if m.get("src") != file_name]
    manifest.append({
        "date": date_str,
        "src": file_name,
        "prompt": prompt,
        "ref": meta["ref"],
        "source": meta.get("source", "unknown")
    })
    save_manifest(manifest_path, manifest)

    log(f"Saved {today_path} (run #{run_num} for {date_str}) and updated latest.jpg"
        + (" [FALLBACK]" if used_fallback else ""))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[generator] FATAL: {e}")
        sys.exit(1)
