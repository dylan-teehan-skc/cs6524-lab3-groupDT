import cv2 as cv
import numpy as np

sift = cv.SIFT_create(contrastThreshold=0.02)  # default only gets ~400 points at this res
PX = 2.0


def load_frames(path, step):
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(frame)
        ok, frame = cap.read()
    return frames[::step]


def detect(frame):
    kp, des = sift.detectAndCompute(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), None)
    return np.float64([k.pt for k in kp]), des


def match(des1, des2):
    knn = cv.BFMatcher().knnMatch(des1, des2, k=2)
    return np.array([(a.queryIdx, a.trainIdx) for a, b in knn if a.distance < 0.75 * b.distance])


def moved(pts1, pts2, m, min_move):
    return np.linalg.norm(pts1[m[:, 0]] - pts2[m[:, 1]], axis=1) >= min_move


# the spin in video b keeps pausing so skip frames where the shoe didnt move
def keyframes(frames, min_move, min_points=50):
    keep = [frames[0]]
    last = detect(frames[0])
    for frame in frames[1:]:
        cur = detect(frame)
        if moved(last[0], cur[0], match(last[1], cur[1]), min_move).sum() >= min_points:
            keep.append(frame)
            last = cur
    return keep


def track(frames, K, min_move):
    feats = [detect(f) for f in frames]
    pts = [p for p, _ in feats]
    tid = [np.full(len(p), -1) for p in pts]
    n = 0
    for i in range(len(frames) - 1):
        m = match(feats[i][1], feats[i + 1][1])
        # for video b the background doesnt move so get rid of it
        m = m[moved(pts[i], pts[i + 1], m, min_move)]
        p1, p2 = pts[i][m[:, 0]], pts[i + 1][m[:, 1]]
        E, mask = cv.findEssentialMat(p1, p2, K, cv.RANSAC, 0.999, 1.0)
        if i == 0:
            _, R, t, _ = cv.recoverPose(E, p1, p2, K, mask=mask.copy())
            first_pose = np.hstack([R, t])
        # using the ransac mask, recoverPose's mask threw away nearly every point
        for a, b in m[mask.ravel() > 0]:
            if tid[i][a] < 0:
                tid[i][a], n = n, n + 1
            tid[i + 1][b] = tid[i][a]
    return pts, tid, n, first_pose


def observations(pts, tid, K):
    f = np.concatenate([np.full((t >= 0).sum(), i) for i, t in enumerate(tid)])
    ids = np.concatenate([t[t >= 0] for t in tid])
    x = np.concatenate([(p[t >= 0] - K[:2, 2]) / K[[0, 1], [0, 1]] for p, t in zip(pts, tid)])
    return f, ids, x


# least squares over every frame a point shows up in
def triangulate(f, ids, x, poses, n, K):
    P = poses[f]
    A = x[:, :, None] * P[:, 2:3] - P[:, :2]
    AtA = np.zeros((n, 4, 4))
    np.add.at(AtA, ids, A.transpose(0, 2, 1) @ A)
    with np.errstate(divide='ignore', invalid='ignore'):
        X = np.linalg.eigh(AtA)[1][:, :, 0]
        X = X[:, :3] / X[:, 3:]
        proj = np.einsum('nij,nj->ni', P, np.c_[X[ids], np.ones(len(ids))])
        err = np.linalg.norm(proj[:, :2] / proj[:, 2:] - x, axis=1) * K[0, 0]
    bad = ~((proj[:, 2] > 0) & (err < PX))
    X[(np.bincount(ids, bad, n) > 0) | (np.bincount(ids, minlength=n) < 2)] = np.nan
    return X, err[~bad].mean()


def visible(X, tid, pts):
    ok = (tid >= 0) & ~np.isnan(X[tid, 0])
    return X[tid[ok]], pts[ok]


def to_pose(rvec, tvec):
    return np.hstack([cv.Rodrigues(rvec)[0], tvec.reshape(3, 1)])


def refine_poses(poses, frames, X, tid, pts, K):
    for k in frames:
        obj, img = visible(X, tid[k], pts[k])
        if len(obj) < 6:
            continue
        rvec = cv.Rodrigues(poses[k, :, :3])[0]
        tvec = poses[k, :, 3:].copy()
        poses[k] = to_pose(*cv.solvePnPRefineLM(obj, img, K, None, rvec, tvec))


def sfm(frames, K, min_move=0):
    pts, tid, n, first_pose = track(frames, K, min_move)
    obs = observations(pts, tid, K)
    poses = np.zeros((len(frames), 3, 4))
    poses[0], poses[1] = np.eye(3, 4), first_pose

    def cloud(m):
        use = obs[0] < m
        return triangulate(*(o[use] for o in obs), poses, n, K)

    X, err = cloud(2)
    print(f'  first two frames: {np.isfinite(X[:, 0]).sum()} points, error {err:.2f}px')
    m = 2
    for i in range(2, len(frames)):
        obj, img = visible(X, tid[i], pts[i])
        if len(obj) < 6:
            break
        # pnp keeps the scale the same, recoverPose only gives the direction of t
        found, rvec, tvec, _ = cv.solvePnPRansac(obj, img, K, None, reprojectionError=PX)
        if not found:
            break
        poses[i], m = to_pose(rvec, tvec), i + 1
        if i % 5 == 0:
            refine_poses(poses, range(max(1, i - 9), m), cloud(m)[0], tid, pts, K)
        X, err = cloud(m)
    refine_poses(poses, range(1, m), X, tid, pts, K)
    X, err = cloud(m)
    return poses[:m], X[~np.isnan(X[:, 0])], err
