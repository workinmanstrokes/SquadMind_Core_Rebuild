# SquadMind Core

High-performance, low-latency AI gaming assistant integrating computer vision, state machine debouncing, tactical voice comms, and controller automation.

## Architecture

1. **Zero-Buffer Video Ingestion (`capture.py`)**:
   Runs in an independent OS thread. Drops accumulated backlog frames to guarantee frame processing is always within milliseconds of real-time console rendering.

2. **Game State & Object Detection (`detector.py`)**:
   Combines YOLOv8 deep learning models with fast OpenCV HSV color masking and Laplacian texture variance to track HUD meters (shot meters, stamina/health, scoreboard).

3. **Finite State Machine (`state_machine.py`)**:
   Implements temporal debouncing and hysteresis so rapid frame flickering or camera shifts do not cause state flapping.

4. **Controller Bridge (`controller_bridge.py`)**:
   Dispatches macros and trigger commands over serial USB directly to hardware adapters (Cronus Zen / Titan Two) or local virtual controller backends.

5. **Tactical Teammate Voice Engine (`voice_engine.py`)**:
   Priority-queued audio/TTS callout engine with anti-spam cooldowns.

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Your Source
Edit `config.yaml` to specify your capture card device index (default `0`), model path, and controller serial port.

### 3. Run
```bash
# Run with live feed:
python3 main.py

# Run for a specific duration in seconds (e.g., 10s test):
python3 main.py 10
```
