# DepthPerception – Vision-Based Smart Home Hazard Detection System

Senior Design — Stony Brook University  
Team: Depth Perception  
Advisor: Prof. Murali Subbarao  
Last Updated:  February 1st 2026

---

## Overview

Depth Perception is a senior design team project at Stony Brook University aimed at developing an assistive safety system for visually impaired individuals navigating indoor environments. The system leverages the Intel RealSense D435i RGB-D camera to identify and localize hazards by combining:

- RGB image analysis  
- Depth sensing and 3D scene reconstruction  
- Motion detection  
- (Planned) Smoke and fire classification using YOLO  
- (Planned) Gas and air-quality sensing via ESP32  
- (Planned) Haptic and audio user feedback mechanisms  

The long-term goal is to create an integrated, multi-modal hazard detection system suitable for home environments.

---

## Current Features (as of 11/27/2025)

### Real-Time RGB, Depth, and IMU Streaming

- Intel RealSense SDK (librealsense2) installed and configured  
- RGB and depth streaming with frame alignment  
- Basic validation and testing using RealSense Viewer  

### Depth Visualization and Scene Reconstruction

- Depth heatmap view for indoor scenes (roughly 0.3–3 m)  
- Observed depth noise beyond ~2.8 m and dropouts on reflective/dark surfaces  
- Validated alignment between RGB and depth frames  

### Near-Obstacle Detection (“Virtual Cane”)

- Threshold-based detection for obstacles within a configurable distance (e.g., 0.8 m)  
- Bounding boxes around close obstacles in the RGB image  
- Approximate distance estimation for closest object  
- “NEAR HAZARD” status overlay with LEFT/CENTER/RIGHT direction cue  

### Motion Detection via Depth Differencing

- Frame-to-frame depth differencing to highlight motion in the scene  
- Binary motion mask plus basic noise filtering and contour detection  

### Multi-Panel Visualization Interface

- 2×2 display layout showing:  
  - RGB frame  
  - Depth heatmap  
  - Near-obstacle mask  
  - Motion mask  

Used for debugging, demos, and progress evaluations.

### Codebase and Repository Setup

- Modular Python code in `src/` with a clear entry point:

  - `main.py` – program entry; launches the demo  
  - `obstacle_detection.py` – RealSense pipeline, obstacle + motion detection  
  - `hazard_stub.py` – placeholder for future YOLO-based smoke/fire detection  

---

## Work in Progress

### YOLO-Based Fire and Smoke Detection

Planned:

- Integrate a lightweight YOLO model for smoke/fire classification  
- Draw bounding boxes and labels on the RGB stream  
- Use depth data to estimate distance to detected hazards  

### ESP32 and Sensor Integration

Planned:

- Use an ESP32 microcontroller for haptic/audio alerts (vibration, buzzer, LEDs)  
- Experiment with gas/air-quality sensors (e.g., MQ-2, BME680)  
- Define a simple protocol between the laptop (RealSense + Python) and ESP32  

---
### Thermal Imaging for Fire Detection (Proof of Concept)

Planned:

- Investigate the use of a low-resolution thermal camera for direct flame and heat-source detection
- Evaluate the FLIR Radiometric Lepton Dev Kit V2 as a potential thermal sensing module
- Use thermal imaging to detect high-temperature regions associated with active fires
- Combine thermal data with RGB-based YOLO detection for improved reliability
- Develop a proof-of-concept pipeline for thermal + depth + RGB sensor fusion

Rationale:

- Thermal imaging enables direct detection of heat signatures that may not be visible in RGB images
- Improves robustness in low-light, smoky, or visually occluded environments
- Provides an independent sensing modality to reduce false positives
- Serves as a validation platform before committing to larger hardware integration

Future Work:

- Interface thermal module with Python processing pipeline
- Calibrate thermal readings against depth and RGB frames
- Evaluate detection accuracy under controlled flame and heating experiments
- Explore integration with ESP32/Raspberry Pi for embedded deployment

---
## Recent Updates (February 2026)

### Audio Feedback Integration

- Added real-time text-to-speech (TTS) audio feedback using `pyttsx3`
- Implemented spoken alerts for:
  - Closest obstacle distance
  - Direction (LEFT / CENTER / RIGHT)
  - Near-hazard status
- Created a dedicated `audio_feedback.py` module to manage speech output

### Debounced Audio Feedback System

- Implemented a cooldown-based “debounce” mechanism to prevent excessive audio output
- Prevents overlapping or rapidly repeated speech when obstacle distance changes quickly
- Uses time-based rate limiting and message caching to improve clarity and usability
  
### Codebase Restructuring

- Retained the `src/` directory for core vision and processing modules  
- Maintained `audio_feedback.py` at the project root for simplified imports and platform compatibility  
- Updated import paths to support mixed root/src organization  
- Reduced setup complexity for collaborators across different development environments  
- Improved project maintainability while avoiding unnecessary packaging overhead


### Technical Challenges and Solutions

#### 1. Text-to-Speech Initialization Errors (Windows)

**Issue:**
- Encountered COM initialization and threading errors when using `pyttsx3` on Windows
- Error: `Cannot change thread mode after it is set`

**Solution:**
- Refactored TTS initialization to occur once at program startup
- Avoided repeated engine reinitialization inside the main loop
- Isolated speech logic inside a dedicated `AudioFeedback` class

#### 2. Audio Overlap and Rapid Speech

**Issue:**
- Distance measurements changed rapidly frame-to-frame
- Resulted in overlapping speech such as:
  “3 meters… 5 meters… 2 meters…”

**Solution:**
- Implemented cooldown timers and message deduplication
- Only allows speech after a minimum time interval
- Prevents repeating identical messages within short intervals
  
### Experimental Person Detection (Prototype)

- Implemented initial person detection using OpenCV HOG + SVM on RGB frames  
- Integrated depth sampling to estimate distance and direction of detected persons  
- Displays bounding boxes and labels for detected individuals  
- Provides preliminary audio feedback for detected persons  
- Currently under testing and optimization due to occasional false positives and missed detections
