import cv2 as cv
import numpy as np
import inspect
from matplotlib import pyplot as plt

### class for experimenting with keypoints, homography, stiching, etc..
class Splitter:
    def __init__(self, img, r_split, c_split, split_size):
        self.img = None
        self.r_split = None
        self.c_split = None
        self.split_size = None
        self.splits = None
        self.split_img(img, r_split, c_split, split_size)

    def __getitem__(self, key):
        return self.splits[key]

    def split_img(self, img, r_split, c_split, split_size):
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
            start_i = i * (m_size // r_split) - (not f_i) * split_size
            end_i = (i + 1) * (m_size // r_split) + (not l_i) * split_size

            for j in range(c_split):
                f_j, l_j = j == 0, j == c_split - 1
                start_j = j * (n_size // c_split) - (not f_j) * split_size
                end_j = (j + 1) * (n_size // c_split) + (not l_j) * split_size

                row.append(
                    img[
                        start_i:end_i + l_i * remain_r,
                        start_j:end_j + l_j * remain_c,
                    ] if img.ndim == 2 else 
                    img[
                        start_i:end_i + l_i * remain_r,
                        start_j:end_j + l_j * remain_c,
                        ::-1
                    ]
                )
                
            splits.append(row)
        print(len(splits), len(splits[0]))
        # self.splits = np.array(splits)

    def draw_splits(self, _lines=None, _rects=None):
        cp_splits = np.copy(self.splits)
        r_split, c_split = cp_splits.shape[0], cp_splits.shape[1]
        c_ss = self.split_size * 2

        for i, row in enumerate(cp_splits):
            for j, img in enumerate(row):
                dx = img.shape[1] - c_ss
                dy = img.shape[0] - c_ss
                
                if j > 0:
                    cv.line(
                        img, (c_ss, 0), (c_ss, len(img)), (255, 0, 0), 2
                    )
                if j < c_split - 1:
                    cv.line(
                        img, (dx, 0), (dx, len(img)), (255, 0, 0), 2
                    )
                if i > 0:
                    cv.line(
                        img, (0, c_ss), (len(img[0]), c_ss), (255, 0, 0), 2
                    )
                if i < r_split - 1:
                    cv.line(
                        img, (0, dy), (len(img[0]), dy), (255, 0, 0), 2
                    )

        return cp_splits

    def show(self, with_lines=False, title=None, **kwargs):
        drawspec_kw = inspect.signature(self.draw_splits)
        showspec_kw = inspect.signature(plt.Axes.imshow)

        cp_splits = self.draw_splits() if with_lines else self.splits
        rows, cols = cp_splits.shape[0], cp_splits.shape[1]
        fig, axes = plt.subplots(
            rows, cols, sharey=True, figsize=(10,10)
        )

        if rows == 1 and cols == 1:
            axes.imshow(cp_splits[0, 0])
            if title:
                axes.set_title(title, **kwargs)
            return

        for i in range(rows):
            for j in range(cols):
                if rows == 1:
                    axes[j].imshow(cp_splits[i, j], **kwargs)
                    axes[j].set_title(f"Split {j}")
                else:
                    axes[i, j].imshow(cp_splits[i, j], **kwargs)
                    axes[i, j].set_title(f"Split {i}, {j}")
