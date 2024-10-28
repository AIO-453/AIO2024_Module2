import numpy as np
import matplotlib.pyplot as plt


class CustomLinearRegression:
    def __init__(self, X_data, y_target, learning_rate=0.01, num_epochs=1000):
        self.num_samples = X_data.shape[0]
        self.X_data = np.c_[np.ones((self.num_samples, 1)), X_data]
        self.y_target = y_target
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Initial weights
        self.theta = np.random.randn(self.X_data.shape[1], 1)
        self.losses = []

    def predict(self, X_data):
        return X_data.dot(self.theta)

    def compute_loss(self, y_pred, y_target):
        loss = 0.5*((y_pred-y_target).T.dot(y_pred-y_target))/self.num_samples
        return loss

    def gradient_descent(self, y_pred):
        grad = self.X_data.T.dot(y_pred-self.y_target)/self.num_samples
        return grad

    def update_weights(self, grad):
        theta = self.theta - self.learning_rate*grad
        return theta

    def fit(self):
        for epoch in range(self.num_epochs):
            # Compute predict
            y_pred = self.predict(self.X_data)

            # Compute loss
            loss = self.compute_loss(y_pred, self.y_target)
            self.losses.append(loss)

            # Compute gradient
            grad = self.gradient_descent(y_pred)

            # Update theta
            self.theta = self.update_weights(grad)

            if (epoch % 50) == 0:
                print(f'Epoch: {epoch} - Loss: {loss}')

        return {
            'loss': self.losses,
            'weight': self.theta
        }
