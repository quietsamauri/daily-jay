# scripts/generate_deepai.py  (Replicate + InstantID version with run suffixes)
import os, datetime, json, pathlib, urllib.request, time, requests, base64, mimetypes

PROMPTS_FILE = "prompts.txt"
OUT_DIR = "daily"
ASSETS_DIR = "assets"
TZ_OFFSET_HOURS = -4  # Rough ET offset for filenames

# Replicate model + version (InstantID). You can update the version from the model's API page if needed.
REPLICATE_MODEL = "zsxkib/instant-id"
REPLICATE_VERSION = "9dcd6d78e7c6560c340d916fe32e9f24aabfa331e5cce95fe31f77fb03121426"

def choose_prompt_and_ref():
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = [p.strip() for p in f if p.strip()]
    if not prompts:
        raise RuntimeError("prompts.txt is empty.")

    # Rotate prompt by day-of-year
    today = datetime.datetime.utcnow() + datetime.timedelta(hours=TZ_OFFSET_HOURS)
    daynum = int(today.strftime("%j"))
    prompt = prompts[daynum % len(prompts)]
    date_str = today.strftime("%Y-%m-%d")

    # Rotate reference image jay1 -> jay2 -> jay3 -> repeat
    refs = sorted([f for f in os.listdir(ASSETS_DIR)
                   if f.lower().startswith("jay") and f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if not refs:
        raise RuntimeError("Put jay1.jpg, jay2.jpg, jay3.jpg into assets/")
    ref = os.path.join(ASSETS_DIR, refs[daynum % len(refs)])
    return prompt, ref, date_str

def replicate_instantid(prompt, api_token, ref_path, width=896, height=1152, steps=28, guidance=6.5):
    """Call Replicate InstantID and return a direct image URL."""
    with open(ref_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    mime = mimetypes.guess_type(ref_path)[0] or "image/jpeg"
    data_url = f"data:{mime};base64,{b64}"

    endpoint = "https://api.replicate.com/v1/predictions"
    headers = {"Authorization": f"Token {api_token}", "Content-Type": "application/json"}
    payload = {
        "version": REPLICATE_VERSION,
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

    r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    prediction = r.json()
    poll_url = prediction["urls"]["get"]

    # Poll until finished
    for _ in range(120):  # ~2 minutes
        pr = requests.get(poll_url, headers=headers, timeout=20)
        pr.raise_for_status()
        p = pr.json()
        if p["status"] == "succeeded":
            out = p["output"]
            if isinstance(out, list) and out:
                return out[-1]
            if isinstance(out, str):
                return out
            raise RuntimeError(f"Unexpected output format: {out}")
        if p["status"] in ("failed", "canceled"):
            raise RuntimeError(f"Replicate failed: {p.get('error') or p['status']}")
        time.sleep(2)

    raise RuntimeError("Replicate timed out")

def download(url, dest, attempts=5):
    for i in range(attempts):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception:
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
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        raise RuntimeError("REPLICATE_API_TOKEN not set (add it as a repo Secret).")

    prompt, ref_image, date_str = choose_prompt_and_ref()
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Build today's base name and add a run suffix (1,2,3...) per day
    base_name = date_str
    existing = [f for f in os.listdir(OUT_DIR)
                if f.startswith(base_name + "-") and f.endswith(".jpg")]
    run_num = len(existing) + 1
    file_name = f"{base_name}-{run_num}.jpg"

    today_path = f"{OUT_DIR}/{file_name}"
    latest_path = f"{OUT_DIR}/latest.jpg"

    meta = {"date": date_str, "prompt": prompt, "ref": os.path.basename(ref_image)}

    try:
        img_url = replicate_instantid(prompt, api_token, ref_image)
        download(img_url, today_path)
        meta["source"] = "replicate-instantid"
    except Exception as e:
        fb = pick_asset_fallback()
        if not fb:
            raise
        with open(fb, "rb") as src, open(today_path, "wb") as dst:
            dst.write(src.read())
        meta.update({"source": "fallback-asset", "error": str(e)})

    # Update latest.jpg for the homepage
    with open(today_path, "rb") as src, open(latest_path, "wb") as dst:
        dst.write(src.read())

    # Write today's meta (used by homepage)
    with open(f"{OUT_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Update manifest for the gallery (append this run)
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

    print(f"Saved {today_path} (run #{run_num} for {date_str}) and updated latest.jpg")

if __name__ == "__main__":
    main()
