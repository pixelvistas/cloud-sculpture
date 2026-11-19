# Cloud Sculpture

A layered acrylic data sculpture of the thunderstorm that crossed southern Ontario on the evening of 2 September 2026, inspired by Dexter Callender's *Light Sculptures* (MIT Media Lab). Cloud-top temperature from the GOES-19 clean infrared band (band 13) is traced every 10 minutes from 18:01 to 23:51 local time, and time becomes height: each acrylic layer is one 20-minute frame of the storm.

The finished piece has 18 layers of 4.5 mm green-edge acrylic, one per 20 minutes from 18:01 to 23:41, and measures 190 × 115 × 81 mm (7.5 × 4.5 × 3.2 in).

## Repository layout

```
goes_extract.py      1. download satellite frames from NOAA's public GOES-19 archive
find_windows.py         (analysis) find time windows where the storm stays wide
inspect_frames.py       (analysis) check edge contact and choose the temperature threshold
build_volume.py      2. build the storm volume (STL) and the visualizer data
slice_layers.py      3. slice the STL into laser-cut layers and nest them on a sheet
requirements.txt

build/sep02_huge/    output of step 2 for the chosen storm frame
cut/makerspace/      output of step 3: the laser-cut files
viewer/              the web visualizer (three.js)
```

## Viewing the visualizer

Browsers do not load `frames.json` from a page opened directly from disk, so serve the folder:

```
cd viewer
python -m http.server 8000
```

Then open `http://localhost:8000`. three.js loads from a CDN; nothing needs to be installed. The acrylic stack beside the contour view is read from `viewer/stacks/`; `stack.name` at the top of `viewer/main.js` selects which one is shown.

## Cutting the sculpture

All files are in `cut/makerspace/`, prepared for a laser bed of 27 × 17 in with red = cut and blue = engrave.

| File | Purpose |
|---|---|
| `test_cut.svg` | Cut first, on an offcut: kerf, engraving depth, guide fit and glue test |
| `sheet_01.svg` | All 18 layers on one 610 × 425 mm panel; **the file for the cutter** |
| `sheet_01_VIEW_ONLY.svg` | Same layout with visible line widths; for checking on screen only |
| `sheet_01_map.png` | Piece IDs, for labelling pieces on the film before removing them |
| `assembly.png` | Stacking order and the time of each layer |
| `report.txt` | Checks: sheet count, weakly guided pieces, stability |

Assemble from layer 1 upward, aligning each piece to the blue guides engraved on the piece below.

## Rebuilding from the raw data

```
pip install -r requirements.txt

python goes_extract.py extract --date 2026-09-02 --start 18:00 --end 23:59 --every 10 \
    --band C13 --bbox -86 38.5 -64 48.5 --out data/2026-09-02_ir_huge

python build_volume.py data/2026-09-02_ir_huge --start 18:01 --end 23:51 \
    --grid-km 3 --sigma-frames 2 --width 630 --out build/sep02_huge

python slice_layers.py build/sep02_huge/sculpture.stl --every 2 --width 200 \
    --thickness 4.5 --sheet 610 425 --base none --colours red-blue \
    --out cut/makerspace --viewer viewer --name makerspace
```

The parameters used for the committed build are recorded in `build/sep02_huge/build_info.json`. After rebuilding, copy `build/sep02_huge/frames.json` and `build/sep02_huge/textures/` into `viewer/data/`.
