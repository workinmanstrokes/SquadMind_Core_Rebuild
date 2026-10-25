import sys
import json
import re
import cv2
import numpy as np

def parse_complex_path_json(raw_output: str):
    try:
        start_idx = raw_output.find("{")
        end_idx = raw_output.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
            return None
        
        json_str = raw_output[start_idx:end_idx + 1]
        data = json.loads(json_str)
        return data
    except Exception as e:
        print(f"[-] Parser Engine Error: {e}")
        return None

def evaluate_hud_metric_mock(frame, norm_x, norm_y, sample_radius=5):
    h, w, _ = frame.shape
    pixel_x = int((norm_x / 1000.0) * w)
    pixel_y = int((norm_y / 1000.0) * h)
    
    pixel_x = max(sample_radius, min(w - sample_radius - 1, pixel_x))
    pixel_y = max(sample_radius, min(h - sample_radius - 1, pixel_y))
    
    roi = frame[pixel_y - sample_radius : pixel_y + sample_radius, 
                pixel_x - sample_radius : pixel_x + sample_radius]
    
    channels_diff = 0
    for bgr_pixel in roi.reshape(-1, 3):
        b, g, r = bgr_pixel
        channels_diff += abs(int(r) - int(g)) + abs(int(g) - int(b))
        
    average_channel_variance = channels_diff / len(roi.reshape(-1, 3))
    return average_channel_variance < 15.0

def run_pipeline_test_suite():
    print("[*] Initializing SquadMind Core Rebuild Verification Engine...")
    errors_detected = False

    print("[*] Running Test 1: Simulating Frame Buffers & Color Metrics...")
    mock_screen = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.circle(mock_screen, (1632, 972), 30, (0, 255, 0), -1)
    
    is_on_cooldown = evaluate_hud_metric_mock(mock_screen, 850, 900)
    if is_on_cooldown:
        print("[-] Test 1 Failed: Local color algorithm marked an active element as desaturated.")
        errors_detected = True
    else:
        print("[+] Test 1 Passed: Local OpenCV metrics correctly identified active color structures.")

    print("[*] Running Test 2: Checking Multi-Step JSON Parser Execution...")
    sample_claude_response = """
    Executing action command.
    ```json
    {
      "executionMode": "SEQUENCE",
      "entryStepId": "step_0",
      "globalTimeoutMs": 3000,
      "metadataReason": "Executing macro sequence over validated OpenCV clusters",
      "steps": [
        { "id": "step_0", "type": "CLICK", "x": 850, "y": 900, "durationMs": 150 }
      ]
    }
    ```
    """
    
    parsed_command = parse_complex_path_json(sample_claude_response)
    if parsed_command is None or parsed_command.get("executionMode") != "SEQUENCE":
        print("[-] Test 2 Failed: Parser failed to correctly parse structural graph payload keys.")
        errors_detected = True
    else:
        steps_list = parsed_command.get("steps", [])
        if steps_list and steps_list[0]["x"] == 850 and steps_list[0]["y"] == 900:
            print("[+] Test 2 Passed: Dynamic coordinate transformations parsed cleanly.")
            print(f"    Reason: '{parsed_command.get('metadataReason')}'")
        else:
            print("[-] Test 2 Failed: Structural node dictionary step attributes mismatched.")
            errors_detected = True

    print("\n==================================================")
    if not errors_detected:
        print("[+] SQUADMIND CORE PIPELINE VERIFICATION SUCCESSFUL: Ready for main loop execution.")
        sys.exit(0)
    else:
        print("[-] PIPELINE VERIFICATION CRITICAL ERROR: Rectify configuration mismatch items.")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline_test_suite()
