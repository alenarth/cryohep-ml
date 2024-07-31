import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Evaluation:
    def report(self, y_true, y_pred):
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.iloc[:, 0]

        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.iloc[:, 0]

        mse, mae, r2 = self.evaluate(y_true, y_pred)

        print(f"Root Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R2 Score: {r2}")

        self.plot_residuals(y_true, y_pred)

    def evaluate(self, y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    def plot_residuals(self, y_true, y_pred):
        slope, intercept, _, _, _ = stats.linregress(y_true, y_pred)
        line = slope * y_true + intercept
        plt.plot(y_true, line)
        plt.plot(y_true, y_true, color="red")
        plt.scatter(y_true, y_pred)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.show()
        plt.close()
