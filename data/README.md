# Dataset Documentation

## Folder Structure

```
data/
├── images/                   # 320x320 JPG images (Main resolution used)
├── gt.csv                    # Ground Truth for evaluation (image_name, Latitude, Longitude)
├── dataset.csv               # Training pool (2,248 samples)
├── test_dataset.csv          # External test set (1,023 samples)
├── metadata/                 # Supporting metadata files
│   ├── raw/                  # GPS metadata CSVs from EXIF
│   ├── corrections/          # Manual GPS correction batches
│   ├── night_holdout.csv     # Night-specific test set (54 samples)
│   └── reference_coords.json # Reference point for coordinate normalization
└── raw_photos/               # Original HEIC files (download separately)
```

## Raw Photos (Optional)

The original HEIC photos are stored on Google Drive to keep the repository lightweight.

**Download link:** [Google Drive - Raw Photos](https://drive.google.com/drive/folders/1P3ePrcjLZNyNx85o62SDG4hXa4oDz908?usp=sharing)

### When do you need raw photos?
- Only if you want to reprocess images at a different resolution.
- Only if you want to add new photos to the dataset.
- The `images/` folder already contains everything needed to run training and inference.

### Setup (if needed)
1. Download the `raw_photos` folder from Google Drive.
2. Place it in `data/raw_photos/`.
3. Run: `python src/data_prep/convert_images.py`.

## Image Sources

| Folder | Description | Count |
|--------|-------------|-------|
| Building28Area | Photos around Building 28 | 189 |
| Building35First | Building 35 first floor | 125 |
| Building35Lower | Building 35 lower level | 58 |
| Building35Upper | Building 35 upper level | 72 |
| Building35Randoms | Various Building 35 views | 99 |
| Bulding32Area | Photos around Building 32 | 188 |
| LibraryArea | Library and surroundings | 518 |
| UnderBuilding26 | Under Building 26 | 215 |
| NightImagesLibraryArea | Night photos (Library) | 112 |
| nightWithout26AndLibrary | Night photos (other areas) | 409 |
| TestPhotos | External test set photos | ~1,100 |
| ProblematicPhotos | High-error samples moved to training | 167 |
| ProblematicPhotos2 | Second batch of high-error samples | 131 |
