import os
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pandas as pd
from PIL import Image, ExifTags
from pillow_heif import register_heif_opener

# dummy comment

# Register HEIC opener with Pillow
register_heif_opener()

def _get_gps_info(gps_ifd: Dict[int, Any]) -> Dict[str, Any]:
    """
    Parses the GPS IFD (integer keys) into a human-readable dict (string keys).
    """
    gps_parsed = {}
    for k, v in gps_ifd.items():
        name = ExifTags.GPSTAGS.get(k, str(k))
        gps_parsed[name] = v
    return gps_parsed


def _rational_to_float(x) -> float:
    """
    Handles tuples like (numerator, denominator) or PIL's rational types.
    """
    try:
        # PIL may give IFDRational which behaves like a number
        return float(x)
    except Exception:
        pass

    if isinstance(x, tuple) and len(x) == 2:
        num, den = x
        if den == 0:
            return 0.0
        return float(num) / float(den)

    # Some formats may nest tuples; fallback
    return float(x)


def _dms_to_deg(dms) -> float:
    """
    Convert degrees/minutes/seconds (often rationals) to decimal degrees.
    dms is usually a tuple of 3 items: (deg, min, sec)
    """
    deg = _rational_to_float(dms[0])
    minute = _rational_to_float(dms[1])
    sec = _rational_to_float(dms[2])
    return deg + minute / 60.0 + sec / 3600.0


def extract_lat_lon(img_path: Path) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """
    Returns (lat, lon, accuracy, datetime_original) if available.
    """
    try:
        with Image.open(img_path) as im:
            exif = im.getexif()
            if not exif:
                return None, None, None, None
            
            # Extract DateTime (try standard tags)
            # 36867 = DateTimeOriginal, 306 = DateTime, 36868 = DateTimeDigitized
            dt = exif.get(36867) or exif.get(306) or exif.get(36868)

            # Extract GPS
            # 34853 = GPSInfo
            gps_info_tag = 34853
            gps = None
            
            # Check if we have GPS info tag
            if gps_info_tag in exif:
                try:
                    # get_ifd returns a dict of {tag_id: value}
                    gps_ifd = exif.get_ifd(gps_info_tag)
                    if gps_ifd:
                        gps = _get_gps_info(gps_ifd)
                except Exception:
                    # If get_ifd fails or tag is malformed
                    pass
            
            if not gps:
                return None, None, None, dt

            lat = gps.get("GPSLatitude")
            lat_ref = gps.get("GPSLatitudeRef")
            lon = gps.get("GPSLongitude")
            lon_ref = gps.get("GPSLongitudeRef")
            
            # Extract Accuracy if available
            # 31 = GPSHPositioningError
            accuracy = gps.get("GPSHPositioningError")
            if accuracy is not None:
                accuracy = _rational_to_float(accuracy)

            if lat is None or lon is None or lat_ref is None or lon_ref is None:
                return None, None, None, dt

            lat_deg = _dms_to_deg(lat)
            lon_deg = _dms_to_deg(lon)

            # Apply hemisphere
            if isinstance(lat_ref, bytes):
                lat_ref = lat_ref.decode(errors="ignore")
            if isinstance(lon_ref, bytes):
                lon_ref = lon_ref.decode(errors="ignore")

            if str(lat_ref).upper().startswith("S"):
                lat_deg = -lat_deg
            if str(lon_ref).upper().startswith("W"):
                lon_deg = -lon_deg

            return lat_deg, lon_deg, accuracy, dt
            
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None, None, None, None


def main():
    folder_name = input("Enter the folder name inside Photos to process: ").strip()
    if not folder_name:
        print("No folder name provided. Exiting.")
        return

    photos_dir = Path("Photos") / folder_name
    out_csv = Path(f"{folder_name}.csv")

    if not photos_dir.exists():
        print(f"Folder not found: {photos_dir}")
        print("Please check the folder name and try again.")
        return

    print(f"Scanning {photos_dir}...")
    
    rows = []
    exts = {".jpg", ".jpeg", ".png", ".heic", ".JPG", ".JPEG", ".HEIC"}

    for p in sorted(photos_dir.rglob("*")):
        if p.suffix not in exts:
            continue
        lat, lon, accuracy, dt = extract_lat_lon(p)
        rows.append({
            "filename": p.name,
            "path": str(p),
            "datetime": dt,
            "lat": lat,
            "lon": lon,
            "gps_accuracy_m": accuracy,
        })

    if not rows:
        print(f"No matching images found in {photos_dir}.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv} (rows={len(df)})")

    # Optional: simple check for missing GPS
    missing_gps = df[df["lat"].isnull()]
    if not missing_gps.empty:
        print(f"Warning: {len(missing_gps)} images have no GPS data.")

if __name__ == "__main__":
    main()
