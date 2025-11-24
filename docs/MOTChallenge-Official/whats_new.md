# Error Analysis Toolkit Usage Guidelines  

## 🌟 What's New (Features)

This branch enhances the standard tracking evaluation by providing powerful visualization and analysis tools. Specifically, it enables the extraction and visualization of tracking errors:
- Error Frame Extraction: Get the exact video frames containing specific tracking errors:
- False Positives (FP): Detections that do not correspond to a ground-truth object.
- False Negatives (FN): Ground-truth objects that were missed by the tracker.
- Error Heatmaps: Generate visual heatmaps for a spatial understanding of performance:
- Ground-Truth (GT), Detection (PRED), False Positives (FP), False Negatives (FN), and ID Switches (IDSW).
- ID Switch Analysis: Extract pairs of frames (before and after) where an object's ID was incorrectly switched by the tracker.
  
## 🛠️ How to Use  
  
### 1. Place your data  
  
Your ground-truth (GT), tracker results, and seqmap files should be set up according to the TrackEval standard. Refer to the [Official TrackEval Data Preparation Guide](https://github.com/thanhtvt/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md#evaluating-on-your-own-data).

For video frame extraction and visualization, you must also place your original videos inside the following structure: 
```
TrackEval/video/<tracker>/<sequence>.mp4
```
- `<tracker>`: Must match a folder name provided in the `--TRACKERS_TO_EVAL` argument.
- `<sequence>`: Must match a sequence name within your selected `--BENCHMARK` folder.

**Note:** If you are running multiple videos/sequences and multiple trackers simultaneously, all relevant data must be placed inside your specified `--BENCHMARK` directory.  
  
### 2. Execute the Script  

Run the evaluation script with the standard arguments, plus the new visualization arguments described below.  

|Argument|Description|Valid Options|   
|---|---|---|  
|--EXTRACTOR|Selects the type of frames to extract.|"FP, FN"|  
|--HEATMAP|Selects the type of heatmap(s) to generate.|"GT, PRED, FP, FN, IDSW"|  
|--ID_SWITCH|Set to True to enable ID switch frame extraction.|True (or omit/False)|  
    
For other basic arguments, please check out this [official readme](https://github.com/thanhtvt/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md#evaluation)
  
This example will run the evaluation and generate FP frames, GT/PRED heatmaps, and ID switch analysis for the `MOT17` benchmark using the `MPNTrack` tracker: (work on [supplied example data](https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip)):  
```   
python scripts/run_mot_challenge.py \
  --BENCHMARK MOT17 \
  --SPLIT_TO_EVAL train \
  --TRACKERS_TO_EVAL MPNTrack \
  --METRICS CLEAR \
  --USE_PARALLEL False \
  --NUM_PARALLEL_CORES 1 \
  --EXTRACTOR FP \
  --HEATMAP GT PRED \
  --ID_SWITCH True  
```  
  
**Some note:**
- `--TRACKER_TO_EVAL` is the name of the folder which is inside `TrackEval\data\trackers\mot_challenge\<YourChallenge>\` folder
- You must include the `CLEAR` metric in the `--METRICS` argument to enable frame extraction and heatmap generation.
- **IMPORTANT:** If you want to evaluate and get frames from multiple videos with multiple trackers **AT THE SAME TIME**, you must place all of your data inside your `--BENCHMARK` directory.
  
## 📊 Output Structure and Naming  

### 1. Data Preparation
When the example script is run, `clear.py` processes the tracking results and creates raw text files within the `boxdetails` folder. These files contain bounding box and ID information necessary for subsequent frame extraction:  
The format varies based on the type of error being logged:  
- **ID Switch Details:** Logs the two IDs involved in the switch across two frames:  
```
<frame> <id1_gt> <id1> <bb1_left> <bb1_top> <bb1_width> <bb1_height> <id2_gt> <id2> <bb2_left> <bb2_top> <bb2_width> <bb2_height> <id3_gt> <id3> . . .
```  
  **Example:** `173 10 353 411 135 399 15 1335 545 49 141`
- **Other Errors (FP/FN/Heatmap):** Logs the frame number and the bounding box coordinates for the relevant objects:  
```
<frame> <bb1_left> <bb1_top> <bb1_width> <bb1_height> <bb2_left> <bb2_top> <bb2_width> <bb2_height> <bb3_left> . . .
```  
**Example:** `469 1759 410 85 259 1707 405 91 258`

### 2. Output Generation
The `trackeval/extract_frame.py` script then reads these text files and uses the original video frames to generate the visualizations, storing them in the `output/<tracker>/<sequence>/` directory. The final sub-folder depends on your command-line arguments:

|Output Folder|Content|Triggered by Argument|  
|---|---|---|  
|`square_images`|Extracted frames containing bounding boxes for FP or FN errors.|`--EXTRACTOR`|  
|`idsw`|Frames captured immediately before and after an ID switch occurred.|`--ID_SWITCH`|  
|`heatmap`|Generated spatial heatmaps (GT, PRED, FP, FN, IDSW).|`--HEATMAP`|
  
#### Image File Convention

The file names make it easy to identify the frame number and tracking IDs involved in the visualization.
- **For `idsw` Folder**: Images are named `X_Y_Z.jpg`:  
  - `X`: The Ground Truth ID.  
  - `Y`: The Frame Number before the ID switch.
  - `Z`: The Frame Number after the ID switch.  

- **For Other Folders (`square_images`, `heatmap`):** Images are named `A.jpg`:
  - `A`: The specific Frame Number from which the image was extracted.

#### ID Switch Sub-folders

Inside the `idsw` folder, two additional sub-folders are created for enhanced analysis:
- `attach`: Contains images that are the horizontal concatenation of two alphabetically adjacent images (e.g., frame Y and frame Z) from the idsw folder, simplifying before/after comparison.
- `bbox_idsw`: Contains cropped bounding boxes extracted from each image in the idsw folder.
---
For the most precise and detailed understanding, refer to the source code itself.
