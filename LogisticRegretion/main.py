from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
from numpy import ndarray
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)


class LogisticRegression:
    def __init__(self, alpha=1e-3, convergence_threshold=1e-4, verbose=0):
        self.alpha = alpha
        self.convergence_threshold = convergence_threshold
        self.theta: ndarray | None = None
        self.verbose = verbose

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _gradient(self, X, y):
        """
        ∇J(theta) = (1/m) * X^T * (sigmoid(theta^T X) - y)
        """
        m = len(y)
        # h_theta(X) = sigmoid(theta^T X)
        hypothesis = self._sigmoid(np.dot(X, self.theta))
        # ∇J(theta) = (1/m) * X^T   * (h_theta(X) - y)
        gradient = np.dot(X.transpose(), (hypothesis - y)) / m
        return gradient

    def fit(self, X, y):
        """
        Алгоритм сошелся когда изменения весов становятся достаточно малыми,
        и модель приблизительно соответствует оптимальным значениям параметров.
        """
        X: ndarray = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.zeros(X.shape[1])
        last_theta = np.copy(self.theta)
        iteration = 0
        while True:
            gradient = self._gradient(X, y)
            self.theta -= self.alpha * gradient
            euclidean_norm = np.linalg.norm(self.theta - last_theta)
            if self.verbose == 1:
                print(
                    f"\titeration: {iteration}\ntheta: {self.theta}\neuclidean_norm: {euclidean_norm}\n"
                )
            if euclidean_norm < self.convergence_threshold:
                print(f"\tIterations {iteration}\n\tEuclidean norm is {euclidean_norm}")
                break
            last_theta = np.copy(self.theta)
            iteration += 1

    def predict(self, X, threshold=0.5):
        """
        Для предсказания подставляем обученные веса в гипотезу и применяем пороговое значение.
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]
        hypothesises = self._sigmoid(np.dot(X, self.theta))
        predictions = (hypothesises > threshold).astype(int)
        return predictions


def preprocess_data(df, test_size=0.3) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    X = StandardScaler().fit_transform(df.drop("Outcome", axis=1))
    y = df["Outcome"]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\tTest Accuracy: {np.mean(y_pred == y_test) * 100:.2f}%")
    # print(f"classification_report\n{classification_report(y_test, y_pred)}")


if __name__ == "__main__":
    df = pd.read_csv("diabetes.csv")
    correlation_matrix = df.corr()
    fig = px.imshow(
        correlation_matrix,
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(title="Correlation Matrix Heatmap", width=800, height=600)
    # fig.show()

    alpha = 1e-3
    convergence_threshold = 1.5e-4
    sgd_model_kwargs = {
        "loss": "log_loss",
        "random_state": RANDOM_STATE,
        "alpha": alpha,
        "l1_ratio": 0,
        "tol": convergence_threshold,
        "eta0": alpha,
    }

    print("1.LogisticRegression")
    train_and_evaluate(
        LogisticRegression(
            alpha=alpha,
            convergence_threshold=convergence_threshold,
        ),
        *preprocess_data(df),
    )

    print("\n2.LogisticRegression with selected features")
    drop_columns = [
        "BloodPressure",
        "SkinThickness",
        "Age",
    ]
    clean_df = df.drop(drop_columns, axis="columns")
    train_and_evaluate(
        LogisticRegression(
            alpha=alpha,
            convergence_threshold=convergence_threshold,
        ),
        *preprocess_data(clean_df),
    )