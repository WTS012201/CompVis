import math
from types import DynamicClassAttribute
from typing import Any, Tuple, Union
from collections.abc import Callable
import cv2 as cv
import numpy as np
import inspect
import copy
from matplotlib import pyplot as plt

class Splitter:
    def __init__(
        self,
        img: np.ndarray,
        r_split: int,
        c_split: int,
        split_size: int
    ):
        self.img = None
        self.r_split, self.c_split, self.split_size = None, None, None
        self.splits, self.mod_splits = None, None

        self.split_img(img, r_split, c_split, split_size)

    def __getitem__(self, key: Union[Tuple, int, slice]):
        if isinstance(key, Union[int, slice]):
            return self.mod_splits[key]
        i, j = key
        return [row[j] for row in self.mod_splits[i]]

    def __setitem__(self, key: Union[Tuple, int, slice], value: list):
        if isinstance(key, Union[int, slice]):
            self.mod_splits[key] = value
            self.r_split = len(self.mod_splits)
            self.c_split = len(self.mod_splits[0])
        else:
            i, j = key
            self.mod_splits[i][j] = value
            self.r_split = len(self.mod_splits)
            self.c_split = len(self.mod_splits[0])
    
    def __iter__(self):
        return iter(self.mod_splits)

    def restore(
        self,
        rrows: Union[range, int] = None,
        rcols: Union[range, int] = None,
    ):      
        if isinstance(rrows, int):
            rrows = [rrows]
        if isinstance(rcols, int):
            rcols = [rcols]

        for i in rrows if rrows else range(len(self.splits)):
            for j in rcols if rcols else range(len(self.splits[0])):
                self.mod_splits[i][j] = copy.deepcopy(self.splits[i][j])

    def split_img(
        self,
        img: np.ndarray,
        r_split: int,
        c_split: int,
        split_size: int
    ):
        self.img = np.copy(img)
        self.r_split = r_split
        self.c_split = c_split
        self.split_size = split_size

        splits = []
        m_size, n_size = img.shape[0], img.shape[1]
        remain_r, remain_c = m_size % r_split, n_size % c_split

        for i in range(r_split):
            row = []
            f_i, l_i = i == 0, i == c_split - 1
            start_i = i * (m_size // r_split) - (not f_i) * split_size // 2
            end_i = (i + 1) * (m_size // r_split) + (not l_i) * split_size // 2

            for j in range(c_split):
                f_j, l_j = j == 0, j == c_split - 1
                start_j = j * (n_size // c_split) - (not f_j) * split_size // 2
                end_j = (j + 1) * (n_size // c_split) + (not l_j) * split_size // 2

                split = (img[
                    start_i:end_i + l_i * remain_r,
                    start_j:end_j + l_j * remain_c,
                ] if img.ndim == 2 else
                img[
                    start_i:end_i + l_i * remain_r,
                    start_j:end_j + l_j * remain_c,
                    ::-1
                ])

                row.append(np.ascontiguousarray(split, dtype=np.uint8))
            splits.append(row)

        self.splits = splits
        self.mod_splits = copy.deepcopy(splits)

    def draw_splits(
        self,
        rects: bool = False,   
        mod_orig: bool = False,
        color: tuple = (255, 0, 0),    
        alpha: float = 0.25     
    ):
        c_ss = self.split_size

        if rects:
            draw = lambda u, x, y, z: \
            cv.rectangle(
                u, (x[0] - c_ss, x[1] - c_ss) if z else x,
                y if z else (y[0] + c_ss, y[1] + c_ss), color, -1
            )  
        else:
            draw = lambda u, x, y, _: \
            cv.line(u, x, y, color, 3)

        for i, row in enumerate(self.mod_splits):
            for j, img in enumerate(row):
                c = np.copy(img)
                dx = c.shape[1] - c_ss
                dy = c.shape[0] - c_ss

                if j > 0:                   # left
                    draw(c, (c_ss, 0), (c_ss, len(c)), True)                    
                if j < self.c_split - 1:    # right
                    draw(c, (dx, 0), (dx, len(c)), False)
                if i > 0:                   # top
                    draw(c, (0, c_ss), (len(c[0]), c_ss), True)
                if i < self.r_split - 1:    # bot
                    draw(c, (0, dy), (len(c[0]), dy), False)
                if rects:
                    c = cv.addWeighted(c, alpha, img, 1 - alpha, 0)
                self.mod_splits[i][j] = c

        if mod_orig:
            self.splits = copy.deepcopy(self.mod_splits)

    def show(
        self,
        rrows: Union[range, int] = None,
        rcols: Union[range, int] = None,
        **kwargs
    ):
        showspec_kw = inspect.signature(plt.Axes.imshow).parameters
        showspec_kw = {k : kwargs[k] for k, v in showspec_kw.items() if k in kwargs}
        
        self.mod_splits = self.mod_splits

        if isinstance(rrows, int):
            rrows = [rrows]
        if isinstance(rcols, int):
            rcols = [rcols]

        rows = len(rrows) if rrows else len(self.mod_splits)
        cols = len(rcols) if rcols else len(self.mod_splits[0])
        _, axes = plt.subplots(rows, cols, figsize=(20,20))

        if rows == 1 and cols == 1:
            axes.imshow(self.mod_splits[0][0])
            if rrows and rcols:
                si, sj = rrows[0], rcols[0]
                axes.set_title(f"Split {si}, {sj}", **showspec_kw)
            return

        if isinstance(rrows, int):
            rrows = [rrows]
        if isinstance(rcols, int):
            rcols = [rcols]

        for ai, si in enumerate(rrows if rrows else range(rows)):
            for aj, sj in enumerate(rcols if rcols else range(cols)):
                if rows == 1:
                    axes[aj].imshow(self.mod_splits[si][sj], **showspec_kw)
                    axes[aj].set_title(f"Split {si}, {sj}", fontsize=15)
                elif cols == 1:
                    axes[ai].imshow(self.mod_splits[si][sj], **showspec_kw)
                    axes[ai].set_title(f"Split {si}, {sj}", fontsize=15)
                else:
                    axes[ai, aj].imshow(self.mod_splits[si][sj], **showspec_kw)
                    axes[ai, aj].set_title(f"Split {si}, {sj}", fontsize=15)
        
    def apply(
        self,
        transform: Callable[[np.ndarray, Any], Tuple[np.ndarray, Any]],
        rrows: Union[range, int] = None,
        rcols: Union[range, int] = None,
        **kwargs
    ):
        t_kw = inspect.signature(transform).parameters
        t_kw = {k : kwargs[k] for k, v in t_kw.items() if k in kwargs}
        accum = []
        
        if isinstance(rrows, int):
            rrows = [rrows]
        if isinstance(rcols, int):
            rcols = [rcols]
        
        for i in rrows if rrows else range(self.r_split):
            for j in rcols if rcols else range(self.c_split):
                try:
                    self.mod_splits[i][j], data = \
                    transform(self.mod_splits[i][j], **t_kw)
                    accum.append(data)        
                except ValueError:
                    self.mod_splits[i][j] = \
                    transform(self.mod_splits[i][j], **t_kw)
                except Exception as e: print(e)

        return accum if len(accum) > 0 else None

    def apply_and_show(
        self,
        transform: Callable[[np.ndarray, Any], Tuple[np.ndarray, Any]],
        rrows: Union[range, int] = None,
        rcols: Union[range, int] = None,
        **kwargs
    ):  
        accum = self.apply(transform, rrows, rcols, **kwargs)
        self.show(rrows, rcols, **kwargs)

        return accum

def shear(img: np.ndarray, limsx: float = 0.25, limsy: float = 0.25):
    sx, sy = 0, 0
    if np.random.randint(0,2):
        low, upp = (0, limsx) if limsx > 0 else (limsx, 0)
        sx = np.random.uniform(low, upp)
    else:
        low, upp = (0, limsy) if limsy > 0 else (limsy, 0)
        sy = np.random.uniform(low, upp)
    (y, x) = img.shape[:2]
    
    M = np.float32([
        [1, sy, abs(sy) * y if sy < 0 else 0],
        [sx, 1, abs(sx) * x if sx < 0 else 0]
    ])
    s = (int(x + abs(sy) * y), int(y + abs(sx) * x))

    img = cv.warpAffine(img, M, s, borderValue=(255,255,255))
    return img

def scale(img: np.ndarray, scale_lim: float = 0.25):
    scale = 1 + np.random.uniform(-scale_lim, scale_lim)
    scale = np.array([scale, scale])
    scale *= img.shape[:2]

    img = cv.resize(img, scale.astype(int)[::-1])
    return img

def rotation(img: np.ndarray, theta_lim: int = 90):
    (y, x) = img.shape[:2]
    c = (x // 2, y // 2)
    theta_lim = np.random.uniform(-theta_lim, theta_lim, 1)[0]
    M = cv.getRotationMatrix2D(c, theta_lim, 1.0)
    
    r = np.deg2rad(theta_lim)
    cx = (abs(np.sin(r) * y) + abs(np.cos(r) * x))
    cy = (abs(np.sin(r) * x) + abs(np.cos(r) * y))
    M[0, 2] += (cx - x) / 2
    M[1, 2] += (cy - y) / 2

    img = cv.warpAffine(
        img, M, (int(cx), int(cy)), borderValue=(255,255,255)
    )
    return img

def trim_padding(img: np.ndarray):
    rows = img.mean(axis = 1).astype(np.uint8)
    rows = (rows == np.full(3, 255, dtype=np.uint8)).sum(axis = 1)
    
    cols = img.mean(axis = 0).astype(np.uint8)
    cols = (cols == np.full(3, 255, dtype=np.uint8)).sum(axis = 1)

    img = img[rows != 3][:, cols != 3]
    return img

def SIFT(
    img, min_s=0, flags=cv.DrawMatchesFlags_DEFAULT, color=-1, SIFT_draw=False
):
    sift = cv.SIFT_create()
    c_img = img.copy()
    
    kp, des = sift.detectAndCompute(c_img, None)
    kp_des = [v for v in zip(kp, des) if v[0].size >= min_s]
    [kp, des] = zip(*kp_des)
    if SIFT_draw:
        img = cv.drawKeypoints(c_img, kp, img, color=(0,255,0),flags=flags)
    else:
        img = c_img

    return img, (kp, np.array(des))

def pad(split1, split2, axis):
    (y1, x1), (y2, x2) = split1.shape[:2], split2.shape[:2]
    if axis == 1:
        if y1 > y2:
            padimg = np.full_like([], 255, shape=(y1, x2, 3), dtype=np.uint8)
            padimg[:y2] = split2
            split2 = padimg
        elif y2 > y1:
            padimg = np.full_like([], 255, shape=(y2, x1, 3), dtype=np.uint8)
            padimg[:y1] = split1
            split1 = padimg
    else:
        if x1 > x2:
            padimg = np.full_like([], 255, shape=(y2, x1, 3), dtype=np.uint8)
            padimg[:, :x2] = split2
            split2 = padimg
        elif x2 > x1:
            padimg = np.full_like([], 255, shape=(y1, x2, 3), dtype=np.uint8)
            padimg[:, :x1] = split1
            split1 = padimg
        
    return split1, split2

def draw_matches(split1, kp1, split2, kp2, matches, axis):
    y, x = split1.shape[:2]
    shift = np.int32([0, y] if axis == 0 else [x, 0])
    res = np.concatenate((split1, split2), axis=axis)
    
    qp = np.int32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1,1,2)
    tp = np.int32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1,1,2)

    # matches idx
    for p1, p2 in zip(qp, tp):
        p2 += shift
        cv.circle(res, p1.reshape(2), 4, (0, 255, 0), 1, cv.LINE_AA)
        cv.circle(res, p2.reshape(2), 4, (0, 255, 0), 1, cv.LINE_AA)
        cv.line(res, p1.reshape(2), p2.reshape(2), (0, 0, 255), 1, cv.LINE_AA)
    return res

def register(sift_data, split1=None, split2=None, axis=0, thresh=0.5):
    if not (split1 is None and split2 is None):
        split1, split2 = pad(split1, split2, axis)
    [(kp1, des1), (kp2, des2)] = sift_data
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    matches = np.array(
        [[m] for m, n in matches if m.distance < thresh * n.distance]
    )
    
    if not matches.shape[0]:
        raise AssertionError("No keypoints found")
    if len(matches[:, 0]) >= 5:
        qp = np.float32(
            [kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1,1,2)
        tp = np.float32(
            [kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1,1,2)
        H, _ = cv.findHomography(qp, tp, cv.RANSAC, 5.0)
    else:
        raise AssertionError("Not enough keypoints. \
            Try applying a different transformation")

    if not (split1 is None and split2 is None):
        res = draw_matches(split1, kp1, split2, kp2, matches, axis)
    else: res = None

    return res, H

def composite(H, split1, split2):
    [y1, x1] = split1.shape[:2]
    [y2, x2] = split2.shape[:2]
    
    qp = np.float32([[0, 0], [0, y2], [x2, y2], [x2, 0]]).reshape(-1,1,2)
    tp = np.float32([[0, 0], [0, y1], [x1, y1], [x1, 0]]).reshape(-1,1,2)
    tp = cv.perspectiveTransform(tp, H)
    
    pts = np.concatenate((qp, tp), axis = 0)    
    H_min = np.int32(pts.min(axis = 0).flatten())
    H_max = np.int32(pts.max(axis = 0).flatten())
    t = np.int32([-H_min[0], -H_min[1]]) 
    Ht = np.array([
        [1, 0, t[0]],
        [0, 1, t[1]],
        [0, 0, 1   ]
    ])

    warped = cv.warpPerspective(
        split1, Ht.dot(H),
        (H_max[0] - H_min[0], H_max[1] - H_min[1]),
        borderValue=(255,255,255)
    )
    
    w_left = warped[t[1]:y2 + t[1], t[0]:x2 + t[0]] & split2
    warp_gray = cv.cvtColor(w_left, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(warp_gray, 0, 255, cv.THRESH_BINARY)
    w_right = cv.bitwise_and(split2, split2, mask = ~mask)
    warped[t[1]:y2 + t[1], t[0]:x2 + t[0]] = w_left + w_right
    
    return warped

def view_matches(splits, axis, **sift_kw):
    splits = copy.deepcopy(splits)
    if axis == 0 or splits.c_split == 1:
        splits[:] = [splits[:, i] for i in range(splits.c_split)]
    r_s, c_s = splits.r_split, splits.c_split

    data = splits.apply(SIFT, **sift_kw)
    data = [data[i * c_s:i * c_s + c_s] for i in range(r_s)]

    if axis: _, axes = plt.subplots(r_s, 1, figsize=(20,20))
    else: _, axes = plt.subplots(1, r_s, figsize=(20,20))

    for i, s_row in enumerate(splits):
        _max = 0
        for j in range(c_s - 1):
            s1, s2 = s_row[j], s_row[j + 1]
            res, _ = register(data[i][j:j + 2], s1, s2, axis)
            if axis:
                s1, s2 = res[:, :s1.shape[1]], res[:, s1.shape[1]:]
            else:
                s1, s2 = res[:s1.shape[0]], res[s1.shape[0]:]
            s_row[j:j + 2] = s1, s2
            if s1.shape[not axis] > s_row[_max].shape[not axis]: _max = j
        row = None
        for j in range(c_s):
            s_row[j], _ = pad(s_row[j], s_row[_max], axis=axis)
            if j == 0:
                row = s_row[j]
                continue
            row = np.concatenate((row, s_row[j]), axis=axis)
        if r_s > 1:
            axes[i].imshow(row)
        else: 
            axes.imshow(row)

def join_axis(splits, axis, **sift_kw):
    comp_splits = copy.deepcopy(splits)
    if axis == 0:
        comp_splits[:] = [splits[:, i] for i in range(splits.c_split)]
    r_s, c_s = comp_splits.r_split, comp_splits.c_split

    for i, s_row in enumerate(comp_splits):
        for j in range(c_s - 1):
            s1, s2 = s_row[0], s_row[j + 1]
            _, d1 = SIFT(s1, **sift_kw)
            _, d2 = SIFT(s2, **sift_kw)
            _, H = register([d1, d2], s1, s2, axis=1)
            s_row[0] = composite(H, s1, s2) 
        comp_splits[i] = [s_row[0]]

    comp_splits.apply(trim_padding)
    return comp_splits

def stack_imgs(imgs):
    stack = None

    for img in imgs:
        (y, x) = img.shape[:2]
        sy = -0.5 
        M = np.float32([
            [1, sy, abs(sy) * y if sy < 0 else 0],
            [0, 1, 0]
        ])
        s = (int(x + abs(sy) * y), y)

        img = cv.warpAffine(img, M, s, borderValue=(255,255,255))
        img = cv.resize(img, (s[0], int(s[1] * abs(sy))))
        shape = (
            int(s[1] * abs(sy) * 0.1),
            s[0],
        )
        if img.ndim == 3:
            shape = (*shape, img.shape[2])
        bot_pad = np.full(
            fill_value=255, shape=shape, dtype=np.uint8
        )
        img = np.concatenate((img, bot_pad), axis=0)

        if stack is None:
            stack = img
        else:
            stack = np.concatenate((img, stack), axis=0) 
    return stack 

def gen_octave(img, n, sigma = 1, k = math.sqrt(2)):
    octave = []

    for i in range(n):
        kernel = cv.getGaussianKernel(5, sigma) 
        img = cv.filter2D(img, -1, kernel)
        octave.append(img)
        sigma *= k
    
    return octave 

def gen_octaves(img, n_octaves, oct_size):
    octaves = []

    for i in range(n_octaves):
        octave = gen_octave(img, oct_size)
        img = cv.resize(img, (img.shape[0] // 2, img.shape[0] // 2))
        octaves.append(octave)
    
    return octaves 

def gen_DoGs(octave):
    # Convert to grayscale before taking the difference
    for i, scale in enumerate(octave):
        octave[i] = cv.cvtColor(scale, cv.COLOR_BGR2GRAY)

    DoGs = []
    for i in range(len(octave ) - 1):
        sub_img = octave[i + 1] - octave[i]
        DoGs.append(sub_img)

    return DoGs

def extrema_detection(DoGs, s_idx):
    # Using neighboring scales
    s_prev, s_curr, s_next = DoGs[s_idx - 1:s_idx + 2]

    candidate_map = np.full_like(s_curr, fill_value=0)
    pad = 1

    # Reference for neighboring points
    neighs = [
        (i,j) for i in range(-pad, pad + 1)
            for j in range(-pad, pad + 1) if i or j
    ]
    candidates = []

    for i in range(pad, s_curr.shape[0] - pad):
        for j in range(pad, s_curr.shape[1] - pad):
            sample = s_curr[i, j]
            for neigh in neighs:
                ni, nj = neigh[0] + i, neigh[1] + j
                maxima = sample > s_curr[ni, nj] \
                    and sample > s_prev[ni, nj] and sample > s_next[ni, nj]
                minima = sample < s_curr[ni, nj] \
                    and sample < s_prev[ni, nj] and sample < s_next[ni, nj]
                if not maxima and not minima:
                    break
            if maxima or minima:
                candidate_map[i, j] = sample 
                candidates.append((j, i, s_idx))

    return candidate_map, candidates

def keypoint_inter(DoGs, cands, max_it=5, con_thresh=0.03):
    refined = []

    for cand in cands:
        (x, y, s) = cand
        in_neigh, outside = False, False

        for i in range(max_it):
            s_prev, s_curr, s_next = DoGs[s - 1:s + 2]
            neighborhood = np.stack([
                s_prev[y - 1:y + 2, x - 1:x + 2],
                s_curr[y - 1:y + 2, x - 1:x + 2],
                s_next[y - 1:y + 2, x - 1:x + 2],
            ]).astype(np.float32) / 255

            # calculate extremum in the second order Taylor expansion
            offset, contrast = second_order_TE_inter(neighborhood)

            in_neigh = abs(offset[0]) < 0.5 
            in_neigh &= abs(offset[1]) < 0.5 
            in_neigh &= abs(offset[2]) < 0.5 

            # stop if the extremum is in the neighborhood
            if in_neigh: break

            # update extremum
            x += int(round(offset[0]))
            y += int(round(offset[1]))
            s += int(round(offset[2]))

            outside = (x < 1 or x >= s_curr.shape[1] - 1) 
            outside |= (y < 1 or y >= s_curr.shape[0] - 1)
            outside |= (s < 1 or s >= len(DoGs) - 1)

            # discard if extremum is outside the image or scale-space
            if outside: break

        if not in_neigh or outside or abs(contrast) < con_thresh:
            continue
        else:
            x += int(round(offset[0]))
            y += int(round(offset[1]))
            s += int(round(offset[2]))
            refined.append((x, y, s, abs(contrast)))

    refined_map = np.full_like(DoGs[0], fill_value=0)
    for p in refined:
        refined_map[p[1], p[0]] = p[3] * 255 
    return refined_map, refined

def second_order_TE_inter(neigh):
    # Compute first deriv (Gradient) in Taylor series on sample kp with adjacent pixels
    dx = (neigh[1, 1, 2] - neigh[1, 1, 0]) / 2
    dy = (neigh[1, 2, 1] - neigh[1, 0, 1]) / 2
    ds = (neigh[2, 1, 1] - neigh[0, 1, 1]) / 2
    grad = np.array([dx, dy, ds])

    # Compute second deriv (Hessian) in Taylor series on sample kp with adjacent pixels
    candidate = neigh[1, 1, 1]
    dxx = neigh[1, 1, 2] - 2 * candidate + neigh[1, 1, 0]
    dyy = neigh[1, 2, 1] - 2 * candidate + neigh[1, 0, 1]
    dss = neigh[2, 1, 1] - 2 * candidate + neigh[0, 1, 1]
    dxy = (neigh[1, 2, 2] - neigh[1, 2, 0] - neigh[1, 0, 2] + neigh[1, 0, 0]) / 4
    dxs = (neigh[2, 1, 2] - neigh[2, 1, 0] - neigh[0, 1, 2] + neigh[0, 1, 0]) / 4
    dys = (neigh[2, 2, 1] - neigh[2, 0, 1] - neigh[0, 2, 1] + neigh[0, 0, 1]) / 4
    hess = np.array([
        [dxx, dxy, dxs], 
        [dxy, dyy, dys],
        [dxs, dys, dss]
    ])

    # Avoid inverting singular matrix
    if np.linalg.det(hess) == 0:
        return np.zeros(shape=3), 0 

    # Solve for extremum and get the contrast at that point
    z_hat = -np.linalg.inv(hess).dot(grad)
    contrast = candidate + grad.T.dot(z_hat) / 2

    return z_hat, contrast


def elim_edge_responses(DoGs, cands, r=10):
    refined = []

    for cand in cands:
        (x, y, s, _) = cand
        neigh = DoGs[s][y - 1:y + 2, x - 1:x + 2].astype(np.float32) / 255

        dxx = neigh[1, 2] - 2 * neigh[1, 1] + neigh[1, 0]
        dyy = neigh[2, 1] - 2 * neigh[1, 1] + neigh[0, 1]
        dxy = (neigh[2, 2] - neigh[2, 0] - neigh[0, 2] + neigh[0, 0]) / 4

        H = np.array([
            [dxx, dxy],
            [dxy, dyy]
        ])
        tr_H, det_H = np.trace(H), np.linalg.det(H)
 
        lhs = r * (tr_H ** 2)
        rhs = ((r + 1) ** 2) * det_H

        if lhs < rhs:
            refined.append(cand)
        
    refined_map = np.full_like(DoGs[0], fill_value=0)
    for p in refined:
        refined_map[p[1], p[0]] = p[3] * 255
    return refined_map, refined


#  convert kps to absolute coordinates
def convert_keypoints(keypoints, o_idx, o_size, sigma, k):
    a_kps = []
    for (x, y, s_idx, contrast) in keypoints:
        # adjusted diameter of the neighborhood 
        sa = sigma * (2 ** (s_idx / float(o_size))) * k ** s_idx 
        # adjust the point
        pa = (x * (2 ** o_idx), y * (2 ** o_idx)) 
        kp = cv.KeyPoint(*pa, size=sa, response=contrast, octave=o_idx)
        a_kps.append(kp)
        
    return a_kps

N_BINS = 36         # number of bins in gradient histogram
LAMBDA_ORI = 1.5    # scale the reach of the gradient distribution
T = 0.80            # threshold for considering local maxima in the gradient orientation histogram
R = 3               # Radius of the gaussian window

def gen_hist(kp, scale, octave, o_idx):
    hist = np.zeros(N_BINS)
    gauss_img = octave[int(kp.size)] / 255
    shape = gauss_img.shape
    rad = int(round(R * scale))

    for y_shift in range(-rad, rad + 1):
        y_r = int(round(kp.pt[1]) / float(2 ** o_idx)) + y_shift
        if y_r < 1 or y_r >= shape[0] - 2:
            continue
        for x_shift in range(-rad, rad + 1):
            x_r = int(round(kp.pt[0]) / float(2 ** o_idx)) + x_shift
            if x_r < 1 or x_r >= shape[1] - 1:
                continue
            # compute gradient magnitudes and orientation from the current patch
            dx = gauss_img[y_r, x_r + 1] - gauss_img[y_r, x_r - 1]
            dy = gauss_img[y_r + 1, x_r] - gauss_img[y_r - 1, x_r]
            mag = math.sqrt(dx ** 2 + dy ** 2)
            ori = np.rad2deg(np.arctan2(dy, dx))

            # compute sample contribution weight and bin index, then place in hist
            con = np.exp(-(x_shift ** 2 + y_shift ** 2) / (2 * scale ** 2))
            bin_idx = int(round(ori * N_BINS / 360.))
            hist[bin_idx % N_BINS] += con * mag
    
    return hist

def smooth_hist(hist):
    smooth = []

    ## Apply 'six times' circular convolution with a three-tap box filter on neighboring bins
    for i in range(N_BINS):
        smooth_mag = 6 * hist[i] + 4 * (hist[i - 1] + hist[(i + 1) % N_BINS])
        smooth_mag += hist[i - 2] + hist[(i + 2) % N_BINS]
        smooth_mag /= 16.
        smooth.append(smooth_mag)

    return smooth

def compute_reference(kp, smooth):
    max_ori = max(smooth)

    # sample window
    band = (smooth > np.roll(smooth, 1), smooth > np.roll(smooth, -1))
    peaks = np.where(band[0] & band[1])[0]

    ori_kps = []
    # Extract the reference orientations
    for p_idx in peaks:
        p_val = smooth[p_idx]
        if p_val < T * max_ori:
            continue
        l_val = smooth[(p_idx - 1) % N_BINS]
        r_val = smooth[(p_idx + 1) % N_BINS]

        # compute the reference index
        norm_p_idx = p_idx + 0.5 * (l_val - r_val)
        norm_p_idx /= l_val - 2 * p_val + r_val
        norm_p_idx %= N_BINS

        # compute the reference orientation
        ori = 360. - norm_p_idx * 360. / N_BINS
        ori_kp = cv.KeyPoint(*kp.pt, kp.size, ori, kp.response, kp.octave)
        ori_kps.append(ori_kp)

    return ori_kps

def compute_orientations(keypoints, octaves, o_idx, k):
    octave = octaves[o_idx]
    kps, hists = [], []

    for kp in keypoints:
        scale = LAMBDA_ORI * kp.size / (2 ** o_idx)

        hist = gen_hist(kp, scale, octave, o_idx)
        smooth = smooth_hist(hist)
        ori_kps = compute_reference(kp, smooth)

        kps += ori_kps
        hists.append((hist, smooth))

    return hists, kps

def show_hist(hist, title):
    degrees = [str(deg * 10) + '-' + str(deg * 10 + 10) for deg in range(36)]
    max_b = max(hist)
    colors = ["green" if h >= max_b * T else "red" for h in hist]

    plt.figure(figsize=(20,10))
    plt.bar(degrees, hist, color=colors)
    plt.title(title, fontsize=20)
    plt.xlabel("Orientation angle", fontsize=16)
    plt.ylabel("Magnitude", fontsize=16)
    plt.xticks(degrees, rotation='vertical')
    plt.show()

def show_patch_structure(
    sample_hists, octaves, sample_kp, sample_oct_idx, sample_scale_idx
):
    scale = LAMBDA_ORI * sample_kp.size / (2 ** sample_oct_idx)
    rad = int(round(R * scale))
    (x, y) = sample_kp.pt
    x, y = int(x), int(y)

    sample_img = octaves[sample_oct_idx][sample_scale_idx].copy()

    shape = sample_img.shape
    left = int(x / float(2 ** sample_oct_idx)) - rad
    left = left if left > 0 else 0
    right = int(x / float(2 ** sample_oct_idx)) + rad
    right = right if right < shape[1] else shape[1] - 1

    top = int(y / float(2 ** sample_oct_idx)) - rad
    top = top if top > 0 else 0
    bot = int(y / float(2 ** sample_oct_idx)) + rad
    bot = bot if bot < shape[1] else shape[1] - 1

    # This is roughly the same as how we computed the angle and grad mags for orientation
    sample_patch = sample_img[top:bot, left:right].copy()
    gradx = cv.Sobel(sample_patch, cv.CV_64F, 1, 0, ksize=1)
    grady = cv.Sobel(sample_patch, cv.CV_64F, 0, 1, ksize=1)
    norm, angle = cv.cartToPolar(gradx, grady, angleInDegrees=True)

    plt.figure(figsize=(20, 20))

    # display the image
    plt.subplot(1,2,1)
    plt.imshow(sample_patch, cmap='gray', origin='lower')
    plt.subplot(1,2,2)
    plt.imshow(norm, cmap='gray', origin='lower')
    q = plt.quiver(gradx, grady, color='blue')
    plt.show()

    sample_img = cv.cvtColor(sample_img, cv.COLOR_GRAY2RGB)
    b_w = 1
    left = left - b_w if left - b_w > 0 else 0
    right = right + b_w if right + b_w < shape[1] else shape[1] - 1
    top = top - b_w if top - b_w > 0 else 0
    bot = bot + b_w if bot + b_w < shape[1] else shape[1] - 1
    cv.rectangle(sample_img, (left, bot), (right, top), (0, 255, 0), b_w * 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(sample_img)