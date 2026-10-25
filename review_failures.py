import os
import sys
import glob
import cv2
import yaml

class FailureReviewer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        ui_cfg = self.config.get("ui_classification", {})
        self.labels = ui_cfg.get("labels", {0: "UNKNOWN"})
        log_cfg = self.config.get("logging", {})
        self.fail_dir = log_cfg.get("fail_storage_dir", "failed_frames")
        
        self.file_list = sorted(glob.glob(os.path.join(self.fail_dir, "fail_conf_*.png")))
        self.current_index = 0

    def launch_review_workspace(self):
        if not self.file_list:
            print(f"[*] Workspace Empty: No failure frames detected inside '{self.fail_dir}/'.")
            return

        print(f"[+] Loaded {len(self.file_list)} logged edge cases.")
        print("    Controls: [D] Next Frame | [A] Previous Frame | [R] Print Labels | [Q] Exit")

        cv2.namedWindow("SquadMind Edge-Case Review Studio", cv2.WINDOW_NORMAL)

        while True:
            file_path = self.file_list[self.current_index]
            frame = cv2.imread(file_path)
            
            if frame is None:
                continue

            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, h - 60), (w, h), (30, 30, 30), -1)
            status_text = f"File {self.current_index + 1}/{len(self.file_list)} | Source: {os.path.basename(file_path)}"
            cv2.putText(frame, status_text, (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("SquadMind Edge-Case Review Studio", frame)
            
            key = cv2.waitKey(0) & 0xFF
            if key == ord('d'):
                if self.current_index < len(self.file_list) - 1:
                    self.current_index += 1
            elif key == ord('a'):
                if self.current_index > 0:
                    self.current_index -= 1
            elif key == ord('r'):
                print(f"\n[*] Active Target Classification Map Reference:")
                for class_id, label_string in self.labels.items():
                    print(f"    ID {class_id} -> {label_string}")
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    reviewer = FailureReviewer()
    reviewer.launch_review_workspace()
