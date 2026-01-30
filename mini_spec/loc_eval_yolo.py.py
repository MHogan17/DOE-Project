import argparse
import csv
import os
from pathlib import Path
from ultralytics import YOLO
import argparse
import csv
import os
from pathlib import Path
from ultralytics import YOLO

def evaluate_model(model_path, data_path, img_size, project_name, run_name):
    # 1. Load the trained model
    print(f"🧠 Loading trained model from: {model_path}")
    model = YOLO(model_path)

    # 2. Run Standard Evaluation (for metrics)
    print(f"🧪 Evaluating model on dataset: {data_path}")
    metrics = model.val(
        conf=0.05,
        iou=0.6,
        data=data_path,
        imgsz=img_size,
        project=project_name,
        name=run_name,
        split='val'
    )
    
    # Define the save path (using the directory YOLO just created)
    save_dir = Path(project_name) / run_name
    csv_path = save_dir / "results.csv"

    # 3. Extract Individual Coordinates
    print(f"📊 Exporting coordinates to {csv_path}...")
    
    # Run prediction on the validation set to get raw results
    # stream=True handles large datasets efficiently
    results = model.predict(source=data_path, imgsz=img_size, conf=0.05, save=False, stream=True)

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(['picture', 'x_center', 'y_center', 'classification_result'])

        for r in results:
            img_name = os.path.basename(r.path)
            
            # Boxes are in xywh format (x_center, y_center, width, height)
            boxes = r.boxes.xywh.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            names = r.names  # Dictionary mapping ID to class name

            for box, cls_id in zip(boxes, classes):
                x_center, y_center, width, height = box
                class_name = names[int(cls_id)]
                
                writer.writerow([img_name, round(float(x_center), 2), round(float(y_center), 2), class_name])

    print(f"✅ Evaluation and CSV export complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model and export coordinates.")
    parser.add_argument('--model', type=str, required=True, help='Path to your trained .pt file.')
    parser.add_argument('--data', type=str, required=True, help='Path to your data configuration (.yaml) file.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for evaluation.')
    parser.add_argument('--project', type=str, default='runs/segment', help='Project name to save results.')
    parser.add_argument('--name', type=str, default='evaluate', help='Name for the evaluation run folder.')
    
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_path=args.data,
        img_size=args.imgsz,
        project_name=args.project,
        run_name=args.name
    )
def evaluate_model(model_path, data_path, img_size, project_name, run_name):
    # 1. Load the trained model
    print(f"?? Loading trained model from: {model_path}")
    model = YOLO(model_path)

    # 2. Run Standard Evaluation (for metrics)
    print(f"🧪 Evaluating model on dataset: {data_path}")
    metrics = model.val(
        conf=0.05,
        iou=0.6,
        data=data_path,
        imgsz=img_size,
        project=project_name,
        name=run_name,
        split='val'
    )
    
    # Define the save path (using the directory YOLO just created)
    save_dir = Path(project_name) / run_name
    csv_path = save_dir / "results.csv"

    # 3. Extract Individual Coordinates
    print(f"📊 Exporting coordinates to {csv_path}...")
    
    # Run prediction on the validation set to get raw results
    # stream=True handles large datasets efficiently
    results = model.predict(source=data_path, imgsz=img_size, conf=0.05, save=False, stream=True)

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(['picture', 'x_center', 'y_center', 'classification_result'])

        for r in results:
            img_name = os.path.basename(r.path)
            
            # Boxes are in xywh format (x_center, y_center, width, height)
            boxes = r.boxes.xywh.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            names = r.names  # Dictionary mapping ID to class name

            for box, cls_id in zip(boxes, classes):
                x_center, y_center, width, height = box
                class_name = names[int(cls_id)]
                
                writer.writerow([img_name, round(float(x_center), 2), round(float(y_center), 2), class_name])

    print(f"✅ Evaluation and CSV export complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model and export coordinates.")
    parser.add_argument('--model', type=str, required=True, help='Path to your trained .pt file.')
    parser.add_argument('--data', type=str, required=True, help='Path to your data configuration (.yaml) file.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for evaluation.')
    parser.add_argument('--project', type=str, default='runs/segment', help='Project name to save results.')
    parser.add_argument('--name', type=str, default='evaluate', help='Name for the evaluation run folder.')
    
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_path=args.data,
        img_size=args.imgsz,
        project_name=args.project,
        run_name=args.name
    )
