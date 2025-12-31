import os
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
from PIL import Image, ExifTags
from pillow_heif import register_heif_opener

# Register HEIC opener with Pillow
register_heif_opener()

def _get_gps_info(gps_ifd: Dict[int, Any]) -> Dict[str, Any]:
    """Parses the GPS IFD (integer keys) into a human-readable dict (string keys)."""
    gps_parsed = {}
    for k, v in gps_ifd.items():
        name = ExifTags.GPSTAGS.get(k, str(k))
        gps_parsed[name] = v
    return gps_parsed


def _rational_to_float(x) -> float:
    """Handles tuples like (numerator, denominator) or PIL's rational types."""
    try:
        return float(x)
    except Exception:
        pass

    if isinstance(x, tuple) and len(x) == 2:
        num, den = x
        if den == 0:
            return 0.0
        return float(num) / float(den)

    return float(x)


def _dms_to_deg(dms) -> float:
    """Convert degrees/minutes/seconds to decimal degrees."""
    deg = _rational_to_float(dms[0])
    minute = _rational_to_float(dms[1])
    sec = _rational_to_float(dms[2])
    return deg + minute / 60.0 + sec / 3600.0


def extract_lat_lon(img_path: Path) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """Returns (lat, lon, accuracy, datetime_original) if available."""
    try:
        with Image.open(img_path) as im:
            exif = im.getexif()
            if not exif:
                return None, None, None, None
            
            # Extract DateTime (try standard tags)
            dt = exif.get(36867) or exif.get(306) or exif.get(36868)

            # Extract GPS
            gps_info_tag = 34853
            gps = None
            
            if gps_info_tag in exif:
                try:
                    gps_ifd = exif.get_ifd(gps_info_tag)
                    if gps_ifd:
                        gps = _get_gps_info(gps_ifd)
                except Exception:
                    pass
            
            if not gps:
                return None, None, None, dt

            lat = gps.get("GPSLatitude")
            lat_ref = gps.get("GPSLatitudeRef")
            lon = gps.get("GPSLongitude")
            lon_ref = gps.get("GPSLongitudeRef")
            
            # Extract Accuracy (Tag 31: GPSHPositioningError)
            accuracy = gps.get("GPSHPositioningError")
            if accuracy is not None:
                accuracy = _rational_to_float(accuracy)

            if lat is None or lon is None or lat_ref is None or lon_ref is None:
                return None, None, None, dt

            lat_deg = _dms_to_deg(lat)
            lon_deg = _dms_to_deg(lon)

            # Apply hemisphere corrections
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


def process_folder(folder_path: Path, output_dir: Path, project_root: Path) -> None:
    """Process a single folder and extract metadata to CSV."""
    print(f"\nProcessing: {folder_path.name}")
    out_csv = output_dir / f"{folder_path.name}.csv"
    
    rows = []
    exts = {".jpg", ".jpeg", ".png", ".heic", ".JPG", ".JPEG", ".HEIC"}

    for p in sorted(folder_path.rglob("*")):
        if p.suffix not in exts:
            continue
        
        lat, lon, accuracy, dt = extract_lat_lon(p)
        
        # Fallback for missing accuracy (common in some metadata)
        if lat is not None and accuracy is None:
            accuracy = 65.0  # Default high uncertainty (meters)

        rows.append({
            "filename": p.name,
            "path": str(p.relative_to(project_root)),  # Clean relative path
            "datetime": dt,
            "lat": lat,
            "lon": lon,
            "gps_accuracy_m": accuracy,
        })

    if not rows:
        print(f"  No images found.")
        return

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    
    valid_gps = df['lat'].notnull().sum()
    print(f"  Saved {out_csv.name} | Images: {len(df)} | Valid GPS: {valid_gps}")


def main(folders: Optional[List[str]] = None):
    """
    Extract GPS metadata from raw photos.
    
    Args:
        folders: Optional list of folder names to process. If None, process all folders.
    
    Usage:
        python extract_metadata.py                              # All folders
        python extract_metadata.py --folders Building26Night LibraryNight  # Only these
    """
    # Robustly find paths relative to this script
    script_dir = Path(__file__).resolve().parent
    # Go up two levels: src/data_prep -> src -> ProjectRoot
    project_root = script_dir.parent.parent
    photos_root = project_root / "data" / "raw_photos"
    output_dir = project_root / "data" / "metadata_raw"

    if not photos_root.exists():
        print(f"Error: Photos folder not found at {photos_root}")
        return
        
    if not output_dir.exists():
        os.makedirs(output_dir)

    # Determine which folders to process
    if folders:
        # Only process specified folders
        folder_paths = []
        for folder_name in folders:
            folder_path = photos_root / folder_name
            if folder_path.exists() and folder_path.is_dir():
                folder_paths.append(folder_path)
            else:
                print(f"Warning: Folder not found: {folder_name}")
        
        if not folder_paths:
            print("No valid folders to process.")
            return
            
        print(f"Processing {len(folder_paths)} specified folder(s)...")
    else:
        # Process all folders (default behavior)
        folder_paths = [f for f in photos_root.iterdir() 
                       if f.is_dir() and not f.name.startswith('.')]
        print(f"Scanning all {len(folder_paths)} folders in {photos_root}...")
    
    # Process each folder
    for folder_path in folder_paths:
        process_folder(folder_path, output_dir, project_root)
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract GPS metadata from raw photos.",
        epilog="Examples:\n"
               "  python extract_metadata.py                    # All folders\n"
               "  python extract_metadata.py --folders Night26  # Only Night26 folder",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--folders", "-f",
        nargs="+",
        help="Specific folder names to process (default: all folders)"
    )
    
    args = parser.parse_args()
    main(folders=args.folders)
