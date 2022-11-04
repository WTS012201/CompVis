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
        if isinstance(key, int) or isinstance(key, slice):
            return self.mod_splits[key]
        i, j = key
        return self.mod_splits[i][j]

    def __setitem__(self, key: Union[Tuple, int, slice], value):
        if isinstance(key, int) or isinstance(key, slice):
            self.mod_splits[key] = value
            return 
        i, j = key
        self.mod_splits[i][j] = value
    
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
        color: bool = False,    # color of line/rect
        alpha: float = 0.25     # opacity of rect
    ):
        c_ss = self.split_size

        if rects:
            draw = lambda u, x, y, z: \
            cv.rectangle(
                u, (x[0] - c_ss, x[1] - c_ss) if z else x, # x <-> y
                y if z else (y[0] + c_ss, y[1] + c_ss),
                color if color else (255, 0, 0, 0.2), -1
            )  
        else:
            draw = lambda u, x, y, _ = None: \
            cv.line(u, x, y, color if color else (255, 0, 0), 3)

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
        
        for i in rrows if rrows else range(len(self.splits)):
            for j in rcols if rcols else range(len(self.splits[0])):
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

    return cv.resize(img, scale.astype(int)[::-1])

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