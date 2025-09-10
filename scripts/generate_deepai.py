import os, datetime, json, pathlib, urllib.request, time, requests

PROMPTS_FILE = "prompts.txt"
OUT_DIR = "daily"
ASSETS_DIR = "assets"
TZ_OFFSET_HOURS = -4  # rough ET offset for filenames

def choose_prompt_and_ref():
    # rotate prompt by day-of-year
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompts = [p.strip() for p in f if p.strip()]
    if not prompts:
        raise RuntimeError("prompts.txt is empty.")

    today = datetime.datetime.utcnow() + datetime.timedelta(hours=TZ_OFFSET_HOURS)
    daynum = int(today.strftime("%j"))
    prompt = prompts[daynum % len(prompts)]
    date_str = today.strftime("%Y-%m-%d")

    # rotate reference image jay1 -> jay2 -> jay3 -> repeat
    refs = sorted([f for f in os.listdir(ASSETS_DIR) if f.lower().startswith("jay")])
    if not refs:
        raise RuntimeError("Put jay1.jpg, jay2.jpg, jay3.jpg into assets/")
    ref = os.path.join(ASSETS_DIR, refs[daynum % len(refs)])
    return prompt, ref, date_str

def deepai_img2img(prompt, api_key, ref_path):
    # Using DeepAI "fast-style-transfer" which accepts an input image + text steer
    url = "https://api.deepai.org/api/fast-style-transfer"
    with open(ref_path, "rb") as f:
        data = {"text": prompt}
        files = {"image": f}
        resp = requests.post(url, data=data, files=files, headers={"Api-Key": api_key}, timeout=120)
    resp.raise_for_status()
    j = resp.json()
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

def main():
    api_key = os.environ.get("DEEPAI_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPAI_API_KEY not set.")

    prompt, ref_image, date_str = choose_prompt_and_ref()
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Using reference: {ref_image}")
    img_url = deepai_img2img(
        prompt + ", ultra-detailed, natural skin tones, high dynamic range, realistic lighting",
        api_key,
        ref_image,
    )

    today_path = f"{OUT_DIR}/{date_str}.jpg"
    latest_path = f"{OUT_DIR}/latest.jpg"

    download(img_url, today_path)
    with open(today_path, "rb") as src, open(latest_path, "wb") as dst:
        dst.write(src.read())

    with open(f"{OUT_DIR}/meta.json", "w", encoding="utf-8") as m:
        json.dump({"date": date_str, "prompt": prompt, "ref": os.path.basename(ref_image)}, m, indent=2)

    print(f"Saved {today_path} and updated latest.jpg")

if __name__ == "__main__":
    main()
