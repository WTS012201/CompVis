import math
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
        rects: bool = False,   # use rects to represent split draw
        mod_orig: bool = False,# modfiy original splits
        color: tuple = (255, 0, 0),    # color of line/rect
        alpha: float = 0.25     # opacity of rect
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
    img, min_s=0, flags=cv.DrawMatchesFlags_DEFAULT, color=-1, SIFT_draw=True
):
    sift = cv.SIFT_create()
    c_img = img.copy()
    
    kp, des = sift.detectAndCompute(c_img, None)
    kp_des = [v for v in zip(kp, des) if v[0].size >= min_s]
    [kp, des] = zip(*kp_des)
    if SIFT_draw:
        img = cv.drawKeypoints(c_img, kp, img, color=color,flags=flags)
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

def extrema_detection(scale_space):
    s_prev, s_curr, s_next = scale_space
    candidates = np.full_like(s_curr, fill_value=0)
    pad = 1
    neighs = [
        (i,j) for i in range(-pad, pad + 1) for j in range(-pad, pad + 1) if i or j
    ]

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
                candidates[i, j] = sample 
    return candidates