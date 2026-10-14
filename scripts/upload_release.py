"""Upload panel files to GitHub Release as binary assets.
Files > 100MB can't be committed directly — use release assets (max 2GB each).
"""
import os, json, requests, time

GH_USER = "KKookie55"
REPO = "KHKT"
GH_PAT = os.environ.get("GH_PAT") or os.environ.get("GITHUB_TOKEN")
if not GH_PAT:
    print("ERROR: Set GH_PAT env var first:  export GH_PAT=ghp_xxxxx")
    sys.exit(1)

FILES_TO_UPLOAD = [
    "/home/z/my-project/download/production/district_panel_v2.parquet",
    "/home/z/my-project/download/production/panel_production.parquet",
    "/home/z/my-project/download/real_data_pilot/firms/modis_vietnam_2018_2023_real.csv",
    "/home/z/my-project/public/data/geo/gadm41_VNM_2.json",
]

TAG = "v9.3-data"
RELEASE_NAME = "Phase D data assets"
RELEASE_NOTES = """# Phase D data assets

Large files (>100MB) for Colab notebook training.

## Files

- `district_panel_v2.parquet` (208 MB): District-level training panel, 964K rows × 94 cols (440 districts × 2191 days, 2018-2023). Built from panel_production.parquet via cell→district area-weighted aggregation.
- `panel_production.parquet` (108 MB): Cell-level training panel, 598K rows × 70 cols (273 cells × 2191 days). Source for district panel.
- `modis_vietnam_2018_2023_real.csv` (1.3 MB): Raw MODIS active fire detections for Vietnam 2018-2023 (17,780 records).
- `gadm41_VNM_2.json` (2.4 MB): GADM Vietnam ADM2 boundaries (710 districts, full-resolution polygons).

## Usage in Colab

```python
!wget -O district_panel_v2.parquet 'https://github.com/KKookie55/KHKT/releases/download/v9.3-data/district_panel_v2.parquet'
!wget -O panel_production.parquet 'https://github.com/KKookie55/KHKT/releases/download/v9.3-data/panel_production.parquet'
!wget -O modis_vietnam_2018_2023_real.csv 'https://github.com/KKookie55/KHKT/releases/download/v9.3-data/modis_vietnam_2018_2023_real.csv'
!wget -O gadm41_VNM_2.json 'https://github.com/KKookie55/KHKT/releases/download/v9.3-data/gadm41_VNM_2.json'
```

## Used by

- Phase D Colab notebook: `scripts/phase_d_colab_train.ipynb`
- Built from: scripts/build_district_panel.py + scripts/enrich_district_panel.py
"""

API = "https://api.github.com"
HEADERS = {
    "Authorization": f"token {GH_PAT}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def main():
    print("=== Upload panel files to GitHub Release ===")

    # 1. Check if release exists, create if not
    print("\n[1] Checking for existing release...")
    r = requests.get(f"{API}/repos/{GH_USER}/{REPO}/releases/tags/{TAG}", headers=HEADERS)
    if r.status_code == 200:
        release = r.json()
        print(f"  ✓ Found existing release: {release['html_url']}")
    elif r.status_code == 404:
        print("  Creating new release...")
        payload = {
            "tag_name": TAG,
            "name": RELEASE_NAME,
            "body": RELEASE_NOTES,
            "draft": False,
            "prerelease": False,
        }
        r = requests.post(f"{API}/repos/{GH_USER}/{REPO}/releases", headers=HEADERS, json=payload)
        r.raise_for_status()
        release = r.json()
        print(f"  ✓ Created release: {release['html_url']}")
    else:
        print(f"  Error: {r.status_code} {r.text[:200]}")
        return

    upload_url_base = release["upload_url"].split("{")[0]  # strip {?name,label}
    print(f"  Upload URL: {upload_url_base}")

    # 2. List existing assets
    print("\n[2] Existing assets:")
    r = requests.get(f"{API}/repos/{GH_USER}/{REPO}/releases/{release['id']}/assets", headers=HEADERS)
    existing = {a["name"]: a["id"] for a in r.json()} if r.status_code == 200 else {}

    # 3. Upload each file
    for file_path in FILES_TO_UPLOAD:
        if not os.path.exists(file_path):
            print(f"\n  ✗ Not found: {file_path}")
            continue
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        print(f"\n[3] Uploading {filename} ({file_size/1024/1024:.1f} MB)...")

        # Delete if exists
        if filename in existing:
            print(f"  Deleting existing asset {filename} (id={existing[filename]})")
            requests.delete(f"{API}/repos/{GH_USER}/{REPO}/releases/assets/{existing[filename]}", headers=HEADERS)

        # Upload
        upload_url = f"{upload_url_base}?name={filename}"
        t0 = time.time()
        with open(file_path, "rb") as f:
            r = requests.post(
                upload_url,
                headers={**HEADERS, "Content-Type": "application/octet-stream"},
                data=f,
            )
        if r.status_code == 201:
            asset = r.json()
            print(f"  ✓ Uploaded in {time.time()-t0:.1f}s")
            print(f"    URL: {asset['browser_download_url']}")
            print(f"    Size: {asset['size']/1024/1024:.1f} MB")
        else:
            print(f"  ✗ Upload failed: {r.status_code} {r.text[:200]}")

    print("\n=== Done ===")
    print(f"Release URL: {release['html_url']}")
    print(f"Download base: https://github.com/{GH_USER}/{REPO}/releases/download/{TAG}/")


if __name__ == "__main__":
    main()
