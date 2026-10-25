import os
import glob
import shutil
import yaml
from ultralytics import YOLO

class YoloRetrainEngine:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        log_cfg = self.config.get("logging", {})
        self.fail_dir = log_cfg.get("fail_storage_dir", "failed_frames")
        self.base_model = self.config.get("model_path", "yolov8n.pt")
        
        ui_cfg = self.config.get("ui_classification", {})
        self.labels = ui_cfg.get("labels", {0: "UNKNOWN"})
        
        self.dataset_dir = "dataset_retrain"

    def build_transient_dataset_layout(self):
        images = glob.glob(os.path.join(self.fail_dir, "*.png"))
        if len(images) < 5:
            print(f"[*] Postponing training: Insufficient frame count ({len(images)}/5 required). Keep logging fails.")
            return False

        if os.path.exists(self.dataset_dir):
            shutil.rmtree(self.dataset_dir)
            
        img_train_dir = os.path.join(self.dataset_dir, "images", "train")
        lbl_train_dir = os.path.join(self.dataset_dir, "labels", "train")
        os.makedirs(img_train_dir, exist_ok=True)
        os.makedirs(lbl_train_dir, exist_ok=True)

        print(f"[*] Building temporary training dataset using {len(images)} logged samples...")
        
        for idx, img_path in enumerate(images):
            base_name = f"edge_case_{idx}"
            dest_img = os.path.join(img_train_dir, f"{base_name}.png")
            shutil.copy(img_path, dest_img)

            dest_lbl = os.path.join(lbl_train_dir, f"{base_name}.txt")
            with open(dest_lbl, "w") as lf:
                lf.write("0 0.50 0.50 0.10 0.10\n")
                
        data_spec = {
            "path": os.path.abspath(self.dataset_dir),
            "train": "images/train",
            "val": "images/train",
            "names": {int(k): v for k, v in self.labels.items()}
        }
        
        with open(os.path.join(self.dataset_dir, "data.yaml"), "w") as f:
            yaml.dump(data_spec, f, default_flow_style=False)
            
        return True

    def execute_training_cycle(self, epochs=10):
        if not self.build_transient_dataset_layout():
            return
            
        print(f"[+] Loading baseline optimization weights: {self.base_model}")
        model = YOLO(self.base_model)
        
        print(f"[*] Starting auto-retraining over localized failures data map...")
        model.train(
            data=os.path.join(self.dataset_dir, "data.yaml"),
            epochs=epochs,
            imgsz=640,
            workers=2,
            project="squadmind_runs",
            name="optimized_core"
        )
        print("[+] Training pass complete. New weight map output saved under squadmind_runs/optimized_core/weights/best.pt")

if __name__ == "__main__":
    engine = YoloRetrainEngine()
    engine.execute_training_cycle(epochs=5)
