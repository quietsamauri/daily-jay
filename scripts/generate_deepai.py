import os, datetime, json, pathlib, urllib.request, urllib.parse, time, random

PROMPTS_FILE = "prompts.txt"
OUT_DIR = "daily"
ASSETS_DIR = "assets"
TZ_OFFSET_HOURS = -4  # rough ET offset for filenames

def choose_prompt():
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = [p.strip() for p in f if p.strip()]
    if not prompts:
        raise RuntimeError("prompts.txt is empty.")
    today = datetime.datetime.utcnow() + datetime.timedelta(hours=TZ_OFFSET_HOURS)
    idx = int(today.strftime("%j")) % len(prompts)  # rotate by day-of-year
    return prompts[idx], today.strftime("%Y-%m-%d")

def deepai_text2img(prompt, api_key):
    url = "https://api.deepai.org/api/text2img"
    data = urllib.parse.urlencode({"text": prompt}).encode("utf-8")
    headers = {"Api-Key": api_key}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req) as resp:
        j = json.loads(resp.read().decode())
    if "output_url" not in j:
        raise RuntimeError(f"DeepAI response missing output_url: {j}")
    return j["output_url"]

def download(url, dest, attempts=5):
    for i in range(attempts):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception:
            time.sleep(1 + i)
    raise RuntimeError("Failed to download image.")

def pick_asset_for_fallback():
    if not os.path.isdir(ASSETS_DIR):
        return None
    refs = [f for f in os.listdir(ASSETS_DIR)
            if f.lower().startswith("jay") and f.lower().endswith((".jpg",".jpeg",".png"))]
    if not refs:
        return None
    # rotate deterministically by day
    daynum = int(datetime.datetime.utcnow().strftime("%j"))
    return os.path.join(ASSETS_DIR, sorted(refs)[daynum % len(refs)])

def load_manifest(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []  # list of entries: {date, src, prompt, ref?, source?}

def save_manifest(path, items):
    # keep newest first
    items_sorted = sorted(items, key=lambda x: x["date"], reverse=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items_sorted, f, indent=2)

def main():
    api_key = os.environ.get("DEEPAI_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPAI_API_KEY not set.")

    prompt, date_str = choose_prompt()
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    today_path = f"{OUT_DIR}/{date_str}.jpg"
    latest_path = f"{OUT_DIR}/latest.jpg"

    steer = ", ultra-detailed, photorealistic, natural skin tones, high dynamic range"
    meta = {"date": date_str, "prompt": prompt}

    try:
        img_url = deepai_text2img(prompt + steer, api_key)
        download(img_url, today_path)
        meta["source"] = "deepai"
    except Exception as e:
        # Fallback to an asset so site never breaks
        fallback = pick_asset_for_fallback()
        if not fallback:
            raise
        with open(fallback, "rb") as src, open(today_path, "wb") as dst:
            dst.write(src.read())
        meta.update({"source": "fallback-asset", "ref": os.path.basename(fallback), "error": str(e)})

    # Update latest.jpg
    with open(today_path, "rb") as src, open(latest_path, "wb") as dst:
        dst.write(src.read())

    # Write today's meta (optional, used on the homepage)
    with open(f"{OUT_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # --- NEW: update manifest with real entries only ---
    manifest_path = f"{OUT_DIR}/manifest.json"
    manifest = load_manifest(manifest_path)

    # If today's entry exists, replace it; otherwise append
    manifest = [m for m in manifest if m.get("date") != date_str]
    manifest.append({
        "date": date_str,
        "src": f"{date_str}.jpg",
        "prompt": prompt,
        **({"ref": meta.get("ref")} if "ref" in meta else {}),
        "source": meta.get("source", "unknown")
    })
    save_manifest(manifest_path, manifest)

    print(f"Saved {today_path} and updated latest.jpg; manifest has {len(manifest)} entries")
