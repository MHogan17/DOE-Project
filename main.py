import sys
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageColor import colormap

import GaussianFitter
from loc_eval_yolo import evaluate_model


if len(sys.argv) < 2:
    print("Usage: python3 main.py <img_directory> <magnification> <pixel_size>")

else:
    directory = sys.argv[1]
    magnification = int(sys.argv[2])
    pixel_size = int(sys.argv[3])

    evaluate_model('mini_spec/train52/weights/best.pt', directory)

    fitter = GaussianFitter.GaussianFitter(magnification, pixel_size)
    fitter.fit(directory)

    pngs = []
    for fname in sorted(os.listdir(directory)):
        if fname == 'ROIs.csv' or fname == 'particles.csv' or fname == 'readme.txt':
            continue
        png = np.array((Image.open(directory + fname)))[:, :, 0]
        pngs.append(png)

    with open(directory + 'particles.csv') as f:
        particles = f.read()
        f.close()
    particles = particles.split('\n')[2:-1]

    for i in range(len(pngs)):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(pngs[i], cmap='gray')
        for row in particles:
            row = row.split(',')
            idx = int(row[1])
            x = float(row[2])
            y = float(row[3])
            s = float(row[4])
            cls = int(row[5])
            crlb = float(row[8])

            if cls == 0:
                color = 'yellow'
            else:
                color = 'red'


            if idx == i:
                circ = plt.Circle((x, y), s, color=color, fill=False)
                ax.add_patch(circ)

                circ = plt.Circle((x, y), crlb, color=color, fill=False)
                ax.add_patch(circ)

        plt.show()


