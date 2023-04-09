import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, FashionMNIST


def plot_images(images, labels=None, ncols=None, cmap="gray", dir="other", name=None):
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(images.shape[0])))
    nrows = int(np.ceil(images.shape[0] / ncols))

    plt.figure(figsize=(2 * ncols, 2 * nrows))

    for idx, image in enumerate(images):
        plt.subplot(nrows, ncols, idx + 1)
        plt.imshow(image, cmap=cmap)

        if labels is not None:
            plt.title(labels[idx])
        plt.axis("off")

    plt.subplots_adjust(wspace=0, hspace=0.2)
    if name is not None:
        os.makedirs(f"plots/{dir}", exist_ok=True)
        plt.savefig(os.path.join(f"plots/{dir}", name + ".png"))
    plt.show()


def get_dt(train_dt, test_dt, return_numpy=False):
    X_train, y_train = train_dt.data, train_dt.targets
    X_test, y_test = test_dt.data, test_dt.targets

    if return_numpy:
        return (
            X_train.numpy(),
            X_test.numpy(),
            y_train.numpy(),
            y_test.numpy(),
        )
    return X_train, X_test, y_train, y_test


def get_sample(X, y, class_size, seed=200):
    np.random.seed(seed)
    classes = np.unique(y)
    choosen_idxs = np.array([], dtype=np.int64)
    for c in classes:
        idxs = np.where(y == c)[0]
        sample_idxs = np.random.choice(idxs, size=class_size, replace=False)
        choosen_idxs = np.r_[choosen_idxs, sample_idxs]
    return X[choosen_idxs, :], y[choosen_idxs]


def get_mnist(**kwargs):
    train_dt = MNIST("./data", download=True, train=True)
    test_dt = MNIST("./data", download=True, train=False)
    return get_dt(train_dt, test_dt, **kwargs)


def get_fashion_mnist(**kwargs):
    train_dt = FashionMNIST("./data", download=True, train=True)
    test_dt = FashionMNIST("./data", download=True, train=False)
    return get_dt(train_dt, test_dt, **kwargs)


def save_np(X, y, name, dir="preprocessed"):
    os.makedirs(f"data/{dir}", exist_ok=True)
    with open(os.path.join("data", dir, name), "wb") as f:
        np.save(f, X)
        np.save(f, y)


def load_np(name, dir="preprocessed"):
    with open(os.path.join("data", dir, name), "rb") as f:
        X = np.load(f)
        y = np.load(f)
    return X, y
