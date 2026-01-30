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
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model_path, data_path, img_size=640, project_name='', run_name='evaluate'):
    # 1. Load the trained model
    print(f"Loading trained model from: {model_path}")
    model = YOLO(model_path)

    # 2. Run Standard Evaluation (for metrics)
    print(f"Evaluating model on dataset: {data_path}")
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
    img_directory = 'validation/images/'

    csv_path = img_directory + "ROIs.csv"

    # 3. Extract Individual Coordinates
    print(f"📊 Exporting coordinates to {csv_path}...")
    
    # Run prediction on the validation set to get raw results
    # stream=True handles large datasets efficiently

    pngs = sorted(os.listdir(img_directory))

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow(['picture', 'x_center', 'y_center', 'confidence', 'classification_result'])

        for j in range(len(pngs)):
            result = model(source=img_directory+pngs[j])
            for r in result:
                for i in range(len(r.boxes.xywh)):
                    x = np.array(r.boxes.xywh[i][0])
                    y = np.array(r.boxes.xywh[i][1])
                    conf = np.array(r.boxes.conf[i])
                    cls = np.array(r.boxes.cls[i])
                    writer.writerow([j, x, y, conf, cls])

    print(f"✅ Evaluation and CSV export complete!")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model and export coordinates.")
#     parser.add_argument('--model', type=str, required=True, help='Path to your trained .pt file.')
#     parser.add_argument('--data', type=str, required=True, help='Path to your data configuration (.yaml) file.')
#     parser.add_argument('--imgsz', type=int, default=640, help='Image size for evaluation.')
#     parser.add_argument('--project', type=str, default='runs/segment', help='Project name to save results.')
#     parser.add_argument('--name', type=str, default='evaluate', help='Name for the evaluation run folder.')
#
#     args = parser.parse_args()
#
#     evaluate_model(
#         model_path=args.model,
#         data_path=args.data,
#         img_size=args.imgsz,
#         project_name=args.project,
#         run_name=args.name
#     )
