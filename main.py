import sys
import GaussianFitter
import BlobDetectionSpherical

if len(sys.argv) < 2:
    print("Usage: python3 main.py <img_directory> <magnification> <pixel_size>")

else:
    try:
        directory = sys.argv[1]
        magnification = int(sys.argv[2])
        pixel_size = int(sys.argv[3])

        BlobDetectionSpherical.find_blobs(directory)

        fitter = GaussianFitter.GaussianFitter(magnification, pixel_size)
        fitter.fit(directory)

    except Exception as e:
        print(e)
        print("Usage: python3 main.py <img_directory>")
