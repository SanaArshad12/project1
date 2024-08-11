#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

using namespace std;

class LinearRegression {
private:
    double learning_rate;
    int iterations;
    vector<double> weights; // Weights for each feature
    double bias; // Intercept term

    // Compute the cost (Mean Squared Error)
    double computeCost(const vector<vector<double>>& X, const vector<double>& Y) {
        int n = X.size();
        double cost = 0;
        for (int i = 0; i < n; ++i) {
            double y_pred = predict(X[i]);
            cost += pow(y_pred - Y[i], 2);
        }
        return cost / (2 * n);
    }

public:
    LinearRegression(double lr, int iter) : learning_rate(lr), iterations(iter), bias(0) {}

    // Train the model using gradient descent
    void fit(const vector<vector<double>>& X, const vector<double>& Y) {
        int n = X.size();   // Number of data points
        int m = X[0].size(); // Number of features

        // Initialize weights
        weights.assign(m, 0);

        for (int iter = 0; iter < iterations; ++iter) {
            vector<double> dW(m, 0); // Gradient for weights
            double db = 0;            // Gradient for bias

            for (int i = 0; i < n; ++i) {
                double y_pred = predict(X[i]);
                double error = y_pred - Y[i];

                // Update gradients
                for (int j = 0; j < m; ++j) {
                    dW[j] += error * X[i][j];
                }
                db += error;
            }

            // Update weights and bias
            for (int j = 0; j < m; ++j) {
                weights[j] -= (learning_rate * dW[j]) / n;
            }
            bias -= (learning_rate * db) / n;

            // Print the cost every 100 iterations
            if (iter % 100 == 0) {
                cout << "Cost at iteration " << iter << ": " << computeCost(X, Y) << endl;
            }
        }
    }

    // Predict the output for a given input
    double predict(const vector<double>& x) const {
        double y_pred = bias;
        for (int j = 0; j < x.size(); ++j) {
            y_pred += weights[j] * x[j];
        }
        return y_pred;
    }

    // Print the model's parameters
    void printParameters() const {
        cout << "Weights: ";
        for (double w : weights) {
            cout << w << " ";
        }
        cout << "\nBias: " << bias << endl;
    }

    // Feature normalization (scaling features between 0 and 1)
    static void normalizeFeatures(vector<vector<double>>& X) {
        int n = X.size();
        int m = X[0].size();

        for (int j = 0; j < m; ++j) {
            double min_val = X[0][j];
            double max_val = X[0][j];
            for (int i = 1; i < n; ++i) {
                if (X[i][j] < min_val) min_val = X[i][j];
                if (X[i][j] > max_val) max_val = X[i][j];
            }
            for (int i = 0; i < n; ++i) {
                X[i][j] = (X[i][j] - min_val) / (max_val - min_val);
            }
        }
    }
};

int main() {
    // Example data with two features (X1, X2) and output Y
    vector<vector<double>> X = {
        {1, 2},
        {2, 3},
        {3, 4},
        {4, 5},
        {5, 6}
    };
    vector<double> Y = { 5, 7, 9, 11, 13 };

    // Normalize features
    LinearRegression::normalizeFeatures(X);

    // Hyperparameters
    double learning_rate = 0.01;
    int iterations = 1000;

    // Create Linear Regression model
    LinearRegression lr(learning_rate, iterations);

    // Train the model
    lr.fit(X, Y);

    // Print the learned parameters
    lr.printParameters();

    // Predict a value
    vector<double> x_new = { 6, 7 };
    double y_pred = lr.predict(x_new);
    cout << "Predicted value for x = [6, 7] is y = " << y_pred << endl;

    return 0;
}
