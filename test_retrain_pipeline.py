import os
import sys
import shutil
import cv2
import numpy as np
import yaml

from retrain_yolo import YoloRetrainEngine

def setup_mock_failure_environment(fail_dir, target_count=5):
    if os.path.exists(fail_dir):
        shutil.rmtree(fail_dir)
    os.makedirs(fail_dir, exist_ok=True)
    
    print(f"[*] Simulating {target_count} raw failure frame inputs...")
    for i in range(target_count):
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(mock_frame, (100, 100), (200, 200), (0, 0, 255), -1)
        file_path = os.path.join(fail_dir, f"fail_conf_0.22_mock_{i}.png")
        cv2.imwrite(file_path, mock_frame)
    print(f"[+] Local test mock frames successfully written into '{fail_dir}/'.")

def run_retrain_test_suite():
    print("==================================================")
    print("[*] Initializing SquadMind Auto-Retrain Verification Suite...")
    print("==================================================")
    
    config_file = "config.yaml"
    if not os.path.exists(config_file):
        print("[-] Verification Error: 'config.yaml' not detected in active directory.")
        sys.exit(1)
        
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        
    log_cfg = config.get("logging", {})
    fail_dir = log_cfg.get("fail_storage_dir", "failed_frames")
    
    backup_dir = "failed_frames_user_backup"
    has_existing_data = os.path.exists(fail_dir)
    if has_existing_data:
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(fail_dir, backup_dir)
        print("[*] User asset safeguard: Backed up live data to 'failed_frames_user_backup/'.")

    pipeline_failed = False
    try:
        setup_mock_failure_environment(fail_dir, target_count=6)
        
        print("[*] Initializing YoloRetrainEngine parsing systems...")
        retrain_engine = YoloRetrainEngine(config_path=config_file)
        
        print("[*] Test Phase 1: Validating transient dataset structure building...")
        dataset_built = retrain_engine.build_transient_dataset_layout()
        
        if not dataset_built:
            print("[-] Test Phase 1 Failed: Dataset builder rejected directory layout structure.")
            pipeline_failed = True
        else:
            print("[+] Test Phase 1 Passed: Training annotations and data.yaml maps created cleanly.")
            
        print("[*] Test Phase 2: Verifying data.yaml compliance schemas...")
        spec_path = os.path.join(retrain_engine.dataset_dir, "data.yaml")
        if not os.path.exists(spec_path):
            print("[-] Test Phase 2 Failed: Target mapping specification configuration file missing.")
            pipeline_failed = True
        else:
            with open(spec_path, 'r') as sf:
                parsed_spec = yaml.safe_load(sf)
                if "names" in parsed_spec and "train" in parsed_spec:
                    print("[+] Test Phase 2 Passed: Dynamic label maps comply with tracking schema definitions.")
                else:
                    print("[-] Test Phase 2 Failed: Missing critical parameter validation blocks.")
                    pipeline_failed = True

    except Exception as e:
        print(f"[-] Pipeline Execution Failure: Caught internal exception -> {e}")
        pipeline_failed = True
        
    finally:
        print("[*] Entering cleaning and workspace restoration sequence...")
        if os.path.exists(fail_dir):
            shutil.rmtree(fail_dir)
        if 'retrain_engine' in locals() and os.path.exists(retrain_engine.dataset_dir):
            shutil.rmtree(retrain_engine.dataset_dir)
            
        if has_existing_data and os.path.exists(backup_dir):
            shutil.copytree(backup_dir, fail_dir)
            shutil.rmtree(backup_dir)
            print("[+] User workspace data seamlessly restored.")
            
    print("\n==================================================")
    if not pipeline_failed:
        print("[+] RETRAIN PIPELINE VALIDATION SUCCESSFUL: Script logic matches deployment schema.")
        sys.exit(0)
    else:
        print("[-] RETRAIN PIPELINE VALIDATION CRITICAL FAILURE: Check script file parsing structures.")
        sys.exit(1)

if __name__ == "__main__":
    run_retrain_test_suite()
