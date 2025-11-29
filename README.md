# DepthPerception – Vision-Based Smart Home Hazard Detection System

Senior Design — Stony Brook University  
Team: Depth Perception  
Advisor: Prof. Murali Subbarao  
Last Updated: November 27, 2025

---

## Overview

DepthPerception is an assistive safety system designed to support visually impaired individuals in navigating indoor environments. The system uses the Intel RealSense D435i RGB-D camera to identify and localize indoor hazards by combining:

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
