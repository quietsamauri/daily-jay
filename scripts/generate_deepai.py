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
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        raise RuntimeError("REPLICATE_API_TOKEN not set (add it as a repo Secret).")

    prompt, ref_image, date_str = choose_prompt_and_ref()
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Build today’s base filename
    base_name = date_str

    # Check how many images already exist today
    existing = [f for f in os.listdir(OUT_DIR) if f.startswith(base_name) and f.endswith(".jpg")]
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

    # update latest.jpg
    with open(today_path, "rb") as src, open(latest_path, "wb") as dst:
        dst.write(src.read())

    # write today’s meta
    with open(f"{OUT_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # update manifest
    manifest_path = f"{OUT_DIR}/manifest.json"
    manifest = load_manifest(manifest_path)
    manifest = [m for m in manifest if m.get("src") != file_name]
    manifest.append({
        "date": date_str,
