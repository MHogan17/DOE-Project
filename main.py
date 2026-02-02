import sys
import GaussianFitter
import BlobDetectionSpherical
# from eval_yolov8 import evaluate_model
from loc_eval_yolo import evaluate_model


if len(sys.argv) < 2:
    print("Usage: python3 main.py <img_directory> <magnification> <pixel_size>")

else:
    directory = sys.argv[1]
    magnification = int(sys.argv[2])
    pixel_size = int(sys.argv[3])

    # Add color classification here
    # Store colored ROIs in ROIs.csv
    # evaluate_model('mini_spec/train50/weights/best.pt', 'validation/data.yaml')

    # BlobDetectionSpherical.find_blobs(directory)
    #
    fitter = GaussianFitter.GaussianFitter(magnification, pixel_size)
    fitter.fit(directory)

