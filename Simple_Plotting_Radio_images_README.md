# Simple Plotting Radio Images

This folder contains a cleaned and documented version of `Simple_Plotting_Radio_images.py`.

Files:

- `Simple_Plotting_Radio_images_clean.py` — cleaned script with helper functions and an example `__main__` usage block.
- `Simple_Plotting_Radio_images_README.md` — this README file.
- `Simple_Plotting_Radio_images_requirements.txt` — minimal Python dependencies.

## How to use
1. Update file paths at the top of the `__main__` block to point to your local FITS files.
2. Run:

```bash
pip install -r Simple_Plotting_Radio_images_requirements.txt
python Simple_Plotting_Radio_images_clean.py
```

## Notes
- The script includes helper functions: `loadfits`, `define_contours`, `crop_image`.
- It reprojects X-ray images to match radio residual images for overlaying contours.
- Adjust contour thresholds and plotting vmin/vmax for your data.
