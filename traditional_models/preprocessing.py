import numpy as np
from skimage.transform import rotate


def generate_additional(X, y, times=2, r=(15, 20)):
    """Generates num additional pictures by rotating existing one by random angle from range r"""
    assert isinstance(X, np.ndarray), "X must be a numpy array"
    X_rotated = np.repeat(X, times, axis=0)
    y_rotated = np.repeat(y, times, axis=0)
    signs = np.random.choice([-1, 1], times * X.shape[0])
    rotations = np.random.sample(times * X.shape[0])
    for idx in range(X_rotated.shape[0]):
        X_rotated[idx, :, :] = (
            rotate(
                X_rotated[idx, :, :],
                signs[idx] * rotations[idx] * (r[1] - r[0]) + r[0],
            )
            * 255.0
        )
    return np.r_[X, X_rotated], np.r_[y, y_rotated]


def drop_unused(X, keep=None):
    """Drops pixels that hold no data - pixels that have same value for all given images in X"""
    assert isinstance(X, np.ndarray), "X must be a numpy array"
    if keep is None:
        keep = np.where((np.max(X, axis=0) != np.min(X, axis=0)))[0]
    return X[:, keep], keep


def to_black_white(X, threshold=50):
    """Sets all pixels either to 0 (if their value is smaller than threshold) or 1"""
    assert isinstance(X, np.ndarray), "X must be a numpy array"
    func = np.vectorize(lambda pixel: 0 if pixel < threshold else 1)
    return func(X)
