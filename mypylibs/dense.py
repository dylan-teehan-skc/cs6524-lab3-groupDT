import cv2 as cv
import numpy as np

# same as lab 2 but more disparities since the shoe is close
sgbm = cv.StereoSGBM_create(numDisparities=96, blockSize=7, P1=8 * 49, P2=32 * 49,
                            uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)
NO_D = np.zeros(5)


def camera_centres(poses):
    return -np.einsum('nji,nj->ni', poses[:, :, :3], poses[:, :, 3])


def relative_pose(Pi, Pj):
    R = Pj[:, :3] @ Pi[:, :3].T
    return R, (Pj[:, 3] - R @ Pi[:, 3]).reshape(3, 1)


def rectify(img1, img2, K, R, t):
    size = img1.shape[1::-1]
    R1, R2, P1, P2 = cv.stereoRectify(K, NO_D, K, NO_D, size, R, t)[:4]
    a = cv.remap(img1, *cv.initUndistortRectifyMap(K, NO_D, R1, P1, size, cv.CV_32FC1), cv.INTER_LINEAR)
    b = cv.remap(img2, *cv.initUndistortRectifyMap(K, NO_D, R2, P2, size, cv.CV_32FC1), cv.INTER_LINEAR)
    return a, b, R1, R2, P1, P2


# video b has a still camera so anything that didnt change between frames is background
def drop_still(img1, img2, thresh=10):
    diff = cv.absdiff(cv.cvtColor(img1, cv.COLOR_BGR2GRAY), cv.cvtColor(img2, cv.COLOR_BGR2GRAY))
    mask = cv.dilate((diff > thresh).astype(np.uint8), np.ones((25, 25)))[:, :, None]
    return img1 * mask, img2 * mask


def pair_points(frames, poses, i, K, still_camera):
    img1, img2 = frames[i], frames[i + 1]
    if still_camera:
        img1, img2 = drop_still(img1, img2)
    R, t = relative_pose(poses[i], poses[i + 1])
    a, b, R1, R2, P1, P2 = rectify(img1, img2, K, R, t)
    if P2[1, 3] != 0:  # sgbm only works on side by side pairs
        return np.zeros((0, 3)), np.zeros((0, 3))
    # same as lab 2, if the camera moved left the second frame is the left one
    left, right, Rrect, k = (b, a, R2, i + 1) if P2[0, 3] > 0 else (a, b, R1, i)
    disp = sgbm.compute(left, right) / 16
    v, u = np.nonzero((disp > 0) & left.any(axis=2))
    z = abs(P2[0, 3]) / disp[v, u]
    X = np.column_stack([(u - P1[0, 2]) * z / P1[0, 0], (v - P1[1, 2]) * z / P1[1, 1], z])
    X = (X @ Rrect - poses[k, :, 3]) @ poses[k, :, :3]
    return X, left[v, u][:, ::-1]


def voxel_average(xyz, rgb, size, min_count=3):
    _, inv, count = np.unique(np.floor(xyz / size), axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(count), 6))
    np.add.at(sums, inv.ravel(), np.hstack([xyz, rgb]))
    avg = sums[count >= min_count] / count[count >= min_count, None]
    return avg[:, :3], avg[:, 3:]


def dense_cloud(frames, K, poses, sparse, still_camera=False):
    centre = np.median(sparse, axis=0)
    radius = np.median(np.linalg.norm(camera_centres(poses) - centre, axis=1))
    pairs = [pair_points(frames, poses, i, K, still_camera) for i in range(len(poses) - 1)]
    xyz, rgb = map(np.vstack, zip(*pairs))
    near = np.linalg.norm(xyz - centre, axis=1) < 0.5 * radius  # cuts out the room
    return voxel_average(xyz[near], rgb[near], radius / 150)


def write_ply(path, xyz, rgb):
    header = (f'ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header')
    np.savetxt(path, np.column_stack([xyz * [1, -1, -1], rgb]), fmt='%.4f %.4f %.4f %d %d %d',
               header=header, comments='')
