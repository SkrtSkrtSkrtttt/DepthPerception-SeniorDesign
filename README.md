# Vision-Based Smart Indoor Hazard Detection & Navigation System

Department of Electrical and Computer Engineering, Stony Brook University  
Team: Depth Perception  
Advisor: Prof. Murali Subbarao  
Repo Author: Naafiul Hossain  

---

## Overview

A real-time multimodal safety system that combines computer vision, depth sensing, and thermal imaging to detect indoor hazards and guide users to a safe exit. The system integrates YOLOv8 object detection, ICP-based mapping, A* path planning, and an embedded ESP32 + FLIR thermal pipeline to enable robust perception in real-world environments.

---

## 🚀 Demo — Full System in Action

<p align="center">
  <a href="https://drive.google.com/file/d/1avbhyp39WIdrwuDVQR6I6T60N_gQK89k/view?usp=sharing">
    <img src="images/demo_thumbnail.png" width="700"/>
  </a>
</p>

<p align="center"><em>Click to watch the full system demo</em></p>

---

## Tech Stack

- **Computer Vision:** YOLOv8 (Ultralytics), OpenCV  
- **Depth Sensing:** Intel RealSense D435i  
- **Thermal Sensing:** FLIR Lepton + ESP32 (FreeRTOS, VoSPI, I2C)  
- **Mapping & Localization:** ICP (Open3D), Occupancy Grid Mapping  
- **Path Planning:** A* Algorithm  
- **Embedded Systems:** ESP32 (C, FreeRTOS)  
- **Languages:** Python, C  

---

## How It Works — Big Picture

The system uses both RGB/depth and thermal sensing to monitor the environment in real time.

Each frame, the system performs four key operations:

1. **Camera Localization** — Estimates camera pose using ICP  
2. **Object Detection** — Identifies hazards, people, and doors using YOLOv8  
3. **Mapping** — Updates a top-down occupancy grid of the environment  
4. **Navigation** — Computes the safest path to an exit using A* and provides spoken directions  

In parallel, a FLIR Lepton thermal camera streams temperature data via an ESP32 running FreeRTOS. Thermal data is processed and fused into the occupancy grid as additional hazard zones.

<p align="center">
  <img src="images/system_architecture.png.png" width="700"/>
</p>
<p align="center"><em>Figure: System Architecture Overview</em></p>

---

## Thermal Sensing System — FLIR Lepton + ESP32

To complement RGB and depth-based perception, the system integrates a thermal sensing pipeline using a FLIR Lepton radiometric thermal camera. Unlike standard cameras, the FLIR Lepton measures infrared radiation to capture temperature data, enabling detection of heat sources not visible in RGB images.

### Hardware Integration

The FLIR Lepton is interfaced with an ESP32 microcontroller running **FreeRTOS**, enabling concurrent handling of acquisition and transmission.

- **I2C (CCI):** Sensor configuration and control  
- **VoSPI:** High-speed thermal frame transfer (80×60 resolution)  

### Data Pipeline

- Frames captured via VoSPI  
- Converted to temperature (°C / °F) using TLinear  
- Hot regions detected via thresholds  
- Data streamed over Wi-Fi (TCP)  
- Parsed by `thermal_receiver.py` and fused into the occupancy grid  

---

### Thermal Imaging Results

<p align="center">
  <img src="images/Screenshot 2026-04-27 141744.png" width="500"/>
</p>
<p align="center"><em>Figure T1: ESP32 and FLIR Lepton wiring and hardware setup.</em></p>

---

<p align="center">
  <img src="images/Screenshot 2026-04-27 141852.png" width="500"/>
</p>
<p align="center"><em>Figure T2: Thermal map of an active stove showing high-temperature regions.</em></p>

---

<p align="center">
  <img src="images/Screenshot 2026-04-27 141937.png" width="500"/>
</p>
<p align="center"><em>Figure T3: Thermal image capturing a human heat signature.</em></p>

---

These results demonstrate that the thermal subsystem provides reliable temperature-based perception, enhancing hazard detection beyond RGB and depth sensing.

---

## Key Terminology

### ICP — Iterative Closest Point
Tracks camera motion by aligning consecutive depth point clouds.

### YOLO — You Only Look Once
Real-time object detection model used to identify hazards and objects.

### A* Pathfinding
Finds the shortest safe route through the occupancy grid.

### Occupancy Grid
2D map representing free space, obstacles, hazards, and exits.

### Point Cloud
3D representation of the environment from depth data.

### Depth Deprojection
Converts 2D pixels + depth into real-world 3D coordinates.

---

## File Reference

### `main.py` — System Orchestrator
The central control loop that coordinates the entire system. Runs continuously once per frame and integrates all subsystems.

**Responsibilities:**
- Captures frames from `camera.py`
- Runs object detection via `detector.py`
- Updates the occupancy grid through `mapper.py`
- Registers hazards and exit locations
- Triggers A* path planning via `planner.py`
- Sends navigation instructions to `navigator.py`
- Manages audio output through `audio_feedback.py`
- Displays RGB feed, depth map, and occupancy grid in real time

---

### `camera.py` — RealSense Interface
Handles all interactions with the Intel RealSense D435i camera.

**Responsibilities:**
- Initializes and configures RGB + depth streams  
- Aligns depth frames with RGB frames  
- Converts depth pixels into 3D coordinates (deprojection)  
- Provides synchronized frame data for processing  

---

### `mapper.py` — Mapping & Localization Engine
Builds and maintains a real-time 2D occupancy grid using depth data.

**Responsibilities:**
- Converts depth frames into 3D point clouds  
- Uses ICP to estimate camera motion between frames  
- Tracks camera pose over time (no encoder required)  
- Projects points into a top-down occupancy grid  
- Classifies cells as free, occupied, hazard, or exit  
- Inflates hazard regions with safety buffers  

---

### `detector.py` — Object Detection Module
Performs real-time perception using multiple detection techniques.

**Responsibilities:**
- Runs YOLOv8 on RGB frames to detect objects and hazards  
- Outputs bounding boxes, confidence scores, and labels  
- Computes depth-based distance and 3D position of detections  
- Detects fire using HSV-based color segmentation  
- Identifies doors using depth-gap heuristics  

---

### `planner.py` — Path Planning Engine
Implements A* pathfinding to compute safe navigation routes.

**Responsibilities:**
- Finds shortest path from user position to exit  
- Avoids hazard and obstacle regions  
- Supports 8-directional movement  
- Uses heuristic-based optimization for efficiency  
- Simplifies paths into key waypoints  

---

### `navigator.py` — Navigation & Instruction Generator
Converts path waypoints into human-readable directions.

**Responsibilities:**
- Calculates direction between consecutive waypoints  
- Groups movements into segments  
- Generates turn-by-turn instructions  
- Formats directions for display and audio output  

---

### `audio_feedback.py` — Text-to-Speech System
Provides real-time spoken feedback to the user.

**Responsibilities:**
- Converts navigation instructions into speech  
- Issues hazard warnings based on proximity and stability  
- Uses a separate thread to avoid blocking the main loop  
- Prevents overlapping speech using cooldown logic  

---

### `thermal_receiver.py` — Thermal Data Processing Module
Handles incoming thermal data from the ESP32.

**Responsibilities:**
- Receives thermal frames over Wi-Fi (TCP)  
- Parses structured packet data  
- Converts raw values into temperature readings  
- Identifies high-temperature regions  
- Stores and provides thermal data for system integration  

---

### `updated_threshold_TLinear_highgain.c` — ESP32 Thermal Firmware
Embedded firmware running on the ESP32 for thermal sensing.

**Responsibilities:**
- Configures FLIR Lepton via I2C (CCI)  
- Captures thermal frames via VoSPI  
- Enables TLinear radiometric mode  
- Processes temperature data and thresholds  
- Packages and transmits data over Wi-Fi  
- Uses FreeRTOS for real-time task scheduling  

---

## Results

The following results demonstrate the system’s ability to detect hazards and guide navigation in real time.

<p align="center">
  <img src="images/Screenshot 2026-04-27 023640.png" width="500"/>
</p>
<p align="center"><em>Figure 1a: Intel RealSense D435i setup.</em></p>

<p align="center">
  <img src="images/Screenshot 2026-04-27 020928.png" width="500"/>
</p>
<p align="center"><em>Figure 1b: ESP32 + FLIR thermal setup.</em></p>

---

<p align="center">
  <img src="images/Screenshot 2026-04-21 012334.png" width="600"/>
</p>
<p align="center"><em>Figure 2: Stove thermal reading (~359°F).</em></p>

---

<p align="center">
  <img src="images/Screenshot 2026-04-19 162340.png" width="600"/>
</p>
<p align="center"><em>Figure 3: Fire detection at 41% confidence.</em></p>

---

<p align="center">
  <img src="images/Screenshot 2026-04-19 170942.png" width="600"/>
</p>
<p align="center"><em>Figure 4: Exit detection with navigation instructions.</em></p>

---

<p align="center">
  <img src="images/Screenshot 2026-04-27 023312.png" width="600"/>
</p>
<p align="center"><em>Figure 5: Detection of household objects using YOLOv8.</em></p>

---

<p align="center">
  <img src="images/Screenshot 2026-04-23 005839.png" width="600"/>
</p>
<p align="center"><em>Figure 6: Person detection with spatial awareness.</em></p>

---

---
## Project Structure

```
src/
├── main.py # Main loop — orchestrates full system
├── camera.py # RealSense D435 capture and depth projection
├── mapper.py # ICP pose tracking + occupancy grid
├── detector.py # YOLOv8 + fire/smoke + door detection
├── planner.py # A* pathfinding on occupancy grid
├── navigator.py # Waypoints → spoken directions
├── audio_feedback.py # Thread-safe TTS with Windows COM fix
├── thermal_receiver.py # TCP receiver for ESP32 thermal stream
└── hazard_stub.py # Legacy stub — no longer used

embedded/
└── updated_threshold_TLinear_highgain.c # ESP32 (FreeRTOS) firmware for FLIR Lepton thermal streaming
```

---

## Installation

```bash
pip install pyrealsense2 opencv-python numpy open3d ultralytics pyttsx3 pythoncom
```

On Linux, also install the TTS engine:
```bash
sudo apt-get install espeak
```

---

## Running the System

```bash
cd src
python main.py
```

On first run, YOLOv8 will automatically download the `yolov8n.pt` model (~6MB). An internet connection is required for this step only.

---

## Display

The window shows three panels side by side:

| Panel | What It Shows |
|-------|--------------|
| Left — RGB feed | Live camera image with detection boxes (red = hazard, blue = person, green = exit) |
| Centre — Depth heatmap | Colour-coded depth map (blue = close, red = far) |
| Right — Grid map | Top-down occupancy map (green = free, dark = obstacle, red = hazard, yellow = exit, orange = planned escape path, white dot = user) |

---

