#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

using namespace std;

class LinearRegression {
private:
    vector<double> weights; // Array con los pesos del modelo
    double bias;            // Término de sesgo
    double learningRate;    // Tasa de aprendizaje
    int epochs;             // Número de épocas de entrenamiento

    vector<double> means;
    vector<double> stdDevs;

public:
    LinearRegression(double lr = 0.05, int ep = 10000) 
        : bias(0.0), learningRate(lr), epochs(ep) {}

    vector<double> getWeights() const { return weights; }
    double getBias() const { return bias; }

    // data_scaling(): Escala los valores del conjunto de datos usando Z-score
    vector<vector<double>> data_scaling(const vector<vector<double>>& X, bool isTraining = true) {
        int nSamples = X.size();
        int nFeatures = X[0].size();
        vector<vector<double>> X_scaled = X;

        if (isTraining) {
            means.assign(nFeatures, 0.0);
            stdDevs.assign(nFeatures, 0.0);

            for (int j = 0; j < nFeatures; ++j) {
                double sum = 0.0;
                for (int i = 0; i < nSamples; ++i) {
                    sum += X[i][j];
                }
                means[j] = sum / nSamples;
            }

            for (int j = 0; j < nFeatures; ++j) {
                double sumSq = 0.0;
                for (int i = 0; i < nSamples; ++i) {
                    sumSq += pow(X[i][j] - means[j], 2);
                }
                stdDevs[j] = sqrt(sumSq / nSamples);
                if (stdDevs[j] == 0.0) stdDevs[j] = 1.0;
            }
        }

        for (int i = 0; i < nSamples; ++i) {
            for (int j = 0; j < nFeatures; ++j) {
                X_scaled[i][j] = (X[i][j] - means[j]) / stdDevs[j];
            }
        }

        return X_scaled;
    }

    // fit(): Entrena el modelo
    void fit(const vector<vector<double>>& X, const vector<double>& y) {
        int nSamples = X.size();
        int nFeatures = X[0].size();

        weights.assign(nFeatures, 0.0);
        bias = 0.0;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            vector<double> dw(nFeatures, 0.0);
            double db = 0.0;

            for (int i = 0; i < nSamples; ++i) {
                double y_pred = bias;
                for (int j = 0; j < nFeatures; ++j) {
                    y_pred += weights[j] * X[i][j];
                }

                double error = y_pred - y[i];

                for (int j = 0; j < nFeatures; ++j) {
                    dw[j] += error * X[i][j];
                }
                db += error;
            }

            for (int j = 0; j < nFeatures; ++j) {
                weights[j] -= learningRate * (2.0 / nSamples) * dw[j];
            }
            bias -= learningRate * (2.0 / nSamples) * db;
        }
    }

    // predict(): Predice las salidas
    vector<double> predict(const vector<vector<double>>& X) const {
        int nSamples = X.size();
        int nFeatures = X[0].size();
        vector<double> predictions(nSamples, 0.0);

        for (int i = 0; i < nSamples; ++i) {
            double y_pred = bias;
            for (int j = 0; j < nFeatures; ++j) {
                y_pred += weights[j] * X[i][j];
            }
            predictions[i] = y_pred;
        }

        return predictions;
    }

    // score(): Calcula los errores y el ajuste R^2
    double score(const vector<double>& y_true, const vector<double>& y_pred) const {
        int n = y_true.size();
        double sumY = 0.0;
        for (double val : y_true) sumY += val;
        double meanY = sumY / n;

        double ssTot = 0.0;
        double ssRes = 0.0;

        for (int i = 0; i < n; ++i) {
            ssTot += pow(y_true[i] - meanY, 2);
            ssRes += pow(y_true[i] - y_pred[i], 2);
        }

        return 1.0 - (ssRes / ssTot);
    }
};

// Función auxiliar para lectura de datasets en formato CSV
bool loadCSV(const string& filename, vector<vector<double>>& X, vector<double>& y) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: No se pudo abrir el archivo " << filename << endl;
        return false;
    }

    string line;
    getline(file, line); // Descartar encabezado

    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string cell;
        vector<double> row;

        while (getline(ss, cell, ',')) {
            if (!cell.empty()) {
                row.push_back(stod(cell));
            }
        }

        if (!row.empty()) {
            y.push_back(row.back());
            row.pop_back();
            X.push_back(row);
        }
    }
    file.close();
    return true;
}

int main() {
    cout << fixed << setprecision(4);
    cout << "==========================================================" << endl;
    cout << "   PRÁCTICA 2 - EVALUACIÓN DE MODELOS DE REGRESIÓN   " << endl;
    cout << "==========================================================" << endl;

    // DATASET 1: Ice_cream_selling_data.csv
    vector<vector<double>> X_ice;
    vector<double> y_ice;

    if (loadCSV("Ice_cream_selling_data.csv", X_ice, y_ice)) {
        cout << "\n[1] DATASET: Ice_cream_selling_data.csv (Simple Linear Regression)" << endl;
        LinearRegression modelIce(0.05, 10000);

        vector<vector<double>> X_ice_scaled = modelIce.data_scaling(X_ice, true);
        modelIce.fit(X_ice_scaled, y_ice);

        cout << "  - weights: " << modelIce.getWeights()[0] << endl;
        cout << "  - bias:    " << modelIce.getBias() << endl;

        vector<double> y_hat = modelIce.predict(X_ice_scaled);
        cout << "  - score (R^2): " << modelIce.score(y_ice, y_hat) << endl;
    }

    // DATASET 2: student_exam_scores.csv
    vector<vector<double>> X_student;
    vector<double> y_student;

    if (loadCSV("student_exam_scores.csv", X_student, y_student)) {
        cout << "\n[2] DATASET: student_exam_scores.csv (Multiple Linear Regression)" << endl;
        LinearRegression modelStudent(0.05, 10000);

        vector<vector<double>> X_student_scaled = modelStudent.data_scaling(X_student, true);
        modelStudent.fit(X_student_scaled, y_student);

        cout << "  - weights: ";
        for (double w : modelStudent.getWeights()) cout << w << " ";
        cout << "\n  - bias:    " << modelStudent.getBias() << endl;

        vector<double> y_hat = modelStudent.predict(X_student_scaled);
        cout << "  - score (R^2): " << modelStudent.score(y_student, y_hat) << endl;
    }

    return 0;
}