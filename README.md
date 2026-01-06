# SET_project_eye-tracking
This repository contains Python scripts for processing eye-tracking data and generating fixation heatmaps and AOI statistics. Raw Tobii Pro Lab data are filtered, weighted by fixation duration, and visualized to analyze gaze strategies in facial emotion recognition.

## Note on Data Privacy
This code is provided **for viewing only** and is not intended to be executed.  
Execution is disabled to prevent any potential re-identification of participants from eye-tracking data.

## Files
**Heatmap.py**  
Input: Tobii Pro Lab TSV eye-tracking data and stimulus images.  
Output: Fixation plots and fixation-based heatmaps overlaid on face stimuli.

**Augen_AOI.py**  
Input: Tobii Pro Lab TSV eye-tracking data with AOI annotations.  
Output: CSV files containing eye vs. face AOI hit counts and gaze-switch statistics by gender and emotion.
