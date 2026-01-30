import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.feature import blob_dog
import tifffile


def highpass(frame):
    A = np.fft.fft2(frame)
    A1 = np.fft.fftshift(A)
    M, N = np.shape(A)
    R = 4
    X = np.linspace(0, N-1, N)
    Y = np.linspace(0, M-1, M)
    X, Y = np.meshgrid(X, Y)
    Cx = N/2
    Cy = M/2

    Hi = 1 - np.exp(-((X - Cx) ** 2 + (Y - Cy) ** 2) / (2 * R) ** 2)
    K = A1 * Hi
    K1 = np.fft.ifftshift(K)
    B = np.fft.ifft2(K1)
    out = B - np.min(np.min(B))

    return np.abs(out)


def uniform_small(frame):
    f1 = frame[0:-1, 0:-1]
    f2 = frame[0:-1, 1::]
    f3 = frame[1::, 0:-1]
    f4 = frame[1::, 1::]
    return f1 + f2 + f3 + f4


def uniform_large(frame):
    f1 = frame[0:-4, 0:-4]
    f2 = frame[0:-4, 2:-2]
    f3 = frame[0:-4, 4::]
    f4 = frame[2:-2, 0:-4]
    f5 = frame[2:-2, 2:-2]
    f6 = frame[2:-2, 4::]
    f7 = frame[4::, 0:-4]
    f8 = frame[4::, 2:-2]
    f9 = frame[4::, 4::]

    return f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9


def find_blobs(dirname):
    n = len(os.listdir(dirname))  # number of frames
    if 'ROIs.csv' in os.listdir(dirname):
        n -= 1
    if 'particles.csv' in os.listdir(dirname):
        n -= 1
    final = []
    fnames = []
    count = 0

    for fname in sorted(os.listdir(dirname)):
        if count >= n:
            break
        if fname == 'ROIs.csv' or fname == 'particles.csv':
            continue
        image = np.array(tifffile.imread(dirname + fname))
        final.append(image)
        fnames.append(fname)
        count += 1
    #
    final = np.asarray(final)
    a = 10
    nx = final.shape[1] - 2*a
    ny = final.shape[2] - 2*a
    r_final = np.zeros((n, nx, ny))
    x = []
    y = []
    idx = []

    for i in range(n):
        r_final[i] = final[i][a:-a, a:-a]
        r_final[i] = highpass(r_final[i])

        temp1 = uniform_small(r_final[i])
        temp2 = uniform_large(temp1)
        temp2 = np.pad(temp2, [(2, 2), (2, 2)], mode='constant', constant_values=[(0, 0), (0, 0)])
        temp = temp2 - temp1
        r_final[i] = np.pad(temp, [(1, 0), (1, 0)], mode='constant', constant_values=[(0, 0), (0, 0)])

        blobs_dog = blob_dog(r_final[i][10: -10, 10: -10], min_sigma=5, max_sigma=20,
                             sigma_ratio=1.5, threshold_rel=0.03, overlap=0.0)
        blobs_dog[:, 2] = blobs_dog[:, 2] * np.sqrt(2)
        y.append(blobs_dog[:, 0])
        x.append(blobs_dog[:, 1])
        idx.append(np.zeros(blobs_dog.shape[0]) + i)

        print(f"\rProgress: {round(i/n * 100, 3)}%", flush=True, end='')
        # Uncomment for plot
        # plt.imshow(r_final[i][10: -10, 10: -10])
        # plt.colorbar()
        # plt.show()
        #
        # x0 = blobs_dog[:, 1]
        # y0 = blobs_dog[:, 0]
        # r0 = blobs_dog[:, 2]
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.imshow(r_final[i][10: -10, 10: -10])
        # for xc, yc, rc in zip(x0, y0, r0):
        #     circ = plt.Circle((xc, yc), rc, fill=False, color='red')
        #     ax.add_patch(circ)
        # plt.show()


    x = np.concatenate(x)
    y = np.concatenate(y)
    idx = np.concatenate(idx)
    ROIlist = np.hstack((idx, x, y))
    ROIlist = np.array(ROIlist.reshape((3, x.shape[0])), dtype='float')
    ROIlist = ROIlist.T

    np.savetxt(dirname + 'ROIs.csv', ROIlist, delimiter=',')

# find_blobs('new beads/')
# with open(dirname + 'ROIs.csv') as f:
#     ROIs = f.read()
#     f.close()
#
# ROIs = ROIs.split('\n')[0:-1]
# for row in ROIs:
#     row = row.split(',')
#     idx = int(float(row[0]))
#     x = int(float(row[1]))
#     y = int(float(row[2]))
#
#     image = tifffile.imread(dirname + fnames[idx])
#     image = correct_image(image, gain, offset)[y:y+40, x:x+40]
#
#     plt.imshow(image)
#     plt.show()
