import math
from typing import Tuple
import numpy as np
import pandas as pd

CLASS_NUM = 10
BATCH_SIZE = 100
EPOCH_NUM = 80
LR = 1e-2


def normalize_data(data: np.ndarray):
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    return (data - data_min) / (data_max - data_min)


def load_data():
    # Dataset from https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    red_wine_data = pd.read_csv(
        "datasets/winequality/winequality-red.csv", sep=";"
    ).values
    white_wine_data = pd.read_csv(
        "datasets/winequality/winequality-white.csv", sep=";"
    ).values

    red_features = red_wine_data[:, :11]
    red_labels = red_wine_data[:, 11]

    white_features = white_wine_data[:, :11]
    white_labels = white_wine_data[:, 11]

    return (normalize_data(red_features), red_labels), (
        normalize_data(white_features),
        white_labels,
    )


def split_dataset(data: Tuple[np.ndarray, np.ndarray]):
    features, labels = data

    nums = features.shape[0]
    end_idx = math.floor(nums * 0.7)

    features_train, features_test = features[:end_idx], features[end_idx:]
    labels_train, labels_test = labels[:end_idx], labels[end_idx:]

    return features_train, labels_train, features_test, labels_test


def data_iter(batch_size, x: np.ndarray, y: np.ndarray):
    nums = x.shape[0]
    indices = list(range(nums))
    np.random.shuffle(indices)

    for i in range(0, nums, batch_size):
        batch_indices = np.array(indices[i : min(i + batch_size, nums)])
        yield x[batch_indices], y[batch_indices]


def softmax(z: np.ndarray):
    z_max = np.max(z)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z)


def onehot_map(label):
    onehot_array = np.zeros(CLASS_NUM)
    onehot_array[int(label)] = 1.0

    return onehot_array


def compute_gradient(x_batch, y_batch, w, b):
    m, _ = x_batch.shape
    dj_dw = np.zeros(w.shape)
    dj_db = np.zeros(b.shape)

    for i in range(m):
        err_i = onehot_map(y_batch[i]) - softmax(np.matmul(x_batch[i], w) + b)
        dj_dw -= x_batch[i].reshape((-1, 1)) * err_i.reshape((1, -1))
        dj_db -= err_i

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(w, b, lr, x_batch, y_batch):
    m = x_batch.shape[0]

    for _ in range(m):
        dj_dw, dj_db = compute_gradient(x_batch, y_batch, w, b)

        w -= lr * dj_dw
        b -= lr * dj_db

    return w, b


def cost_function(x_batch, y_batch, w, b):
    m = x_batch.shape[0]

    cost = 0
    for i in range(m):
        cost += -np.dot(
            onehot_map(y_batch[i]), np.log(softmax(np.matmul(x_batch[i], w) + b))
        )

    return cost / m


def train(features: np.ndarray, labels: np.ndarray, w, b, epoch_num):
    cost_history = []

    print(f"training for {epoch_num} times...")
    for epoch in range(epoch_num):
        for x_batch, y_batch in data_iter(BATCH_SIZE, features, labels):
            w, b = gradient_descent(w, b, LR, x_batch, y_batch)
            cost = cost_function(x_batch, y_batch, w, b)
            cost_history.append(cost)

        if (epoch + 1) % 10 == 0:
            x_valid, y_valid = next(data_iter(BATCH_SIZE, features, labels))

            m = x_valid.shape[0]
            predict = np.matmul(x_valid, w) + b
            count = 0
            for i in range(m):
                result = np.argmax(softmax(predict[i]))

                if result == y_valid[i]:
                    count += 1

            print(
                f"epoch [{epoch + 1}/{EPOCH_NUM}] cost:{cost: .6f} accuracy:{count * 100 / m: .6f}"
            )

    return w, b


def test(feature: np.ndarray, labels: np.ndarray, w: np.ndarray, b):
    nums = feature.shape[0]

    predict = np.matmul(feature, w) + b
    count = 0
    for i in range(nums):
        result = np.argmax(softmax(predict[i]))

        if result == labels[i]:
            count += 1

    print(f"accuracy: {count * 100 / nums}%")


def main():
    red_wine, white_wine = load_data()

    (
        red_features_train,
        red_labels_train,
        red_features_test,
        red_labels_test,
    ) = split_dataset(red_wine)

    red_w_init = np.zeros((red_features_train.shape[1], CLASS_NUM))
    red_b_init = np.zeros(CLASS_NUM)

    red_w, red_b = train(
        red_features_train, red_labels_train, red_w_init, red_b_init, EPOCH_NUM
    )

    test(red_features_test, red_labels_test, red_w, red_b)

    np.savez("models/model.npz", red_w=red_w, red_b=red_b)

    # (
    #     white_features_train,
    #     white_labels_train,
    #     white_features_test,
    #     white_labels_test,
    # ) = split_dataset(white_wine)

    # white_w_init = np.zeros((white_features_train.shape[1], CLASS_NUM))
    # white_b_init = np.zeros(CLASS_NUM)

    # white_w, white_b = train(
    #     white_features_train, white_labels_train, white_w_init, white_b_init, EPOCH_NUM
    # )

    # test(white_features_test, white_labels_test, white_w, white_b)


if __name__ == "__main__":
    main()
