#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <list>

using namespace std;

vector<string> headerNames;
vector<vector<double>> features;
vector<double> result;  

vector<double> weights;  

vector<double> leastSquares(vector<vector<double>> feature, vector<double> target);
vector<vector<double>> matrixMult(vector<vector<double>> matrixA, vector<vector<double>> matrixB);
vector<vector<double>> inverseMatrix(vector<vector<double>> matrix);
double determinant(vector<vector<double>> matrix);
vector<vector<double>> getMinor(vector<vector<double>> matrix, int row, int col);
double predict(vector<double> weighted, vector<double> input);
vector<vector<double>> cofactor(vector<vector<double>> matrix);
double meanSquareErr(vector<double> predict, vector<double> actual);
void pullData(string fileName);
vector<vector<double>> transposeMatrix(vector<vector<double>> matrix);

int main() {
    string fileName;
    cout << "Enter the training data file name: ";
    cin >> fileName;

    pullData("/File Path/Machine-Learning/LeastSquares/" + fileName + ".csv");
    
    weights = leastSquares(features, result);

    //Display weights
    cout << "\nLinear Regression Weights:\n";
    for (size_t i = 0; i < weights.size(); ++i) {
        if (i == 0) {
            cout << "Bias: " << weights[i] << "\n";
        } else {
            cout << "Weight for feature " << headerNames[i-1] << ": " << weights[i] << "\n";
        }
    }

    //Create the weights to predict orginial values
    vector<double> prediction;
    for (int i = 0; i < features.size(); i++) {
        double predictor = predict(weights, features[i]);
        prediction.push_back(predictor);
    }

    //Showing the error rate
    cout << "\nMean Square Error: " + to_string(meanSquareErr(prediction, result)*100) + "%";

    //Creating own values
    cout << "\n\nCreate prediction of your final exame score: \n";

    vector<double> input = {1.0};
    for (int i = 0; i < headerNames.size()-1; i++) {
        double val;
        cout << "What is your " + headerNames[i] + ": ";
        cin >> val;
        input.push_back(val);
    }

    cout << "\n\nYour final exam score is predicted to be: " + to_string(predict(weights, input));
    return 0;
}

//This generates the line of best fit for given data
//using the least square formula - (QT * Q)^-1 * (QT * b)
vector<double> leastSquares(
    vector<vector<double>> feature,
    vector<double> target) {
    //QT
    vector<vector<double>> transposeMat = transposeMatrix(feature);

    //(QT*Q)^-1
    vector<vector<double>> xtx = matrixMult(transposeMat, feature);
    vector<vector<double>> inverseMat = inverseMatrix(xtx);


    //QT*b
    vector<vector<double>> columnTarget;
    for (double val : target) {
        columnTarget.push_back({val});
    }
    vector<vector<double>> xty = matrixMult(transposeMat, columnTarget);

    //(QT * Q)^-1 * (QT * b)
    vector<vector<double>> bestFit = matrixMult(inverseMat, xty);

    //Flattening to 1D
    vector<double> weight;
    for (vector<double> row : bestFit) {
        weight.push_back(row[0]);
    }

    return weight;
}

//Multiplies the transpose with the normal matrix
vector<vector<double>> matrixMult(vector<vector<double>> matrixA,
                                    vector<vector<double>> matrixB) {

    if (matrixA[0].size() != matrixB.size()) {
        cerr << "Matrices are incompatible for multiplication";
    }

    int rows = matrixA.size();
    int col = matrixB[0].size();

    //Creating the resulting matrix
    vector<vector<double>> matrix(rows, vector<double>(col));

    for (int i = 0; i < rows; i++) {
        for (int j = 0 ; j < col; j++) {
            for (int k = 0; k < matrixA[0].size(); k++) {
                matrix[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    return matrix;
}

vector<vector<double>> inverseMatrix(vector<vector<double>> matrix) {
    double det = determinant(matrix);

    if (det == 0.0) {
        cerr << "The determinant is 0";
    }

    //Generating the cofactor matrix and then getting the transpose matrix
    int n = matrix.size();
    vector<vector<double>> cof = cofactor(matrix);
    vector<vector<double>> adj = transposeMatrix(cof);

    vector<vector<double>> inverse(n, vector<double>(n));

    //Creating the inverse matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[i][j] = adj[i][j] / det;
        }
    }

    return inverse;
}

//Uses recursion and calls the minor to calculate the determinant
double determinant(vector<vector<double>> matrix) {
    int n = matrix.size();

    if (n == 1) {
        return matrix[0][0];
    }
    if (n == 2) {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    }

    double det = 0.0;

    for (int j = 0; j < n; j++) {
        vector<vector<double>> minor = getMinor(matrix, 0, j);  
        double cofactor = 1.0;
        if (j % 2 != 0) {
            cofactor = -1.0;
        }
        det += cofactor * matrix[0][j] * determinant(minor);
    }

    return det;
}

//This generates the minor for the given
vector<vector<double>> getMinor(vector<vector<double>> matrix, int row, int col) {
    vector<vector<double>> minor;
    for (int i = 0; i < matrix.size(); i++) {
        if (!(i == row)) {
            vector<double> newRow;

            for (int j = 0; j < matrix[0].size(); j++) {
                if (!(j == col)) {
                    newRow.push_back(matrix[i][j]);
                }
            }
            minor.push_back(newRow);
        }
    }
    return minor;
}

//This generates the cofactor matrix
vector<vector<double>> cofactor(vector<vector<double>> matrix) {
    int n = matrix.size();

    vector<vector<double>> cofactor(n, vector<double>(n));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            vector<vector<double>> minor = getMinor(matrix, i, j);
            cofactor[i][j] = pow(-1, i + j) * determinant(minor);
        }
    }

    return cofactor;
}

//Uses the weights to predict
double predict(vector<double> weighted, vector<double> input) {
    double output = 0.0;
    for (int i = 0; i < weights.size(); ++i) {
        output += weights[i] * input[i];
    }
    return output;

}

//Calculates the standard error rate
double meanSquareErr(vector<double> predict, vector<double> actual) {
    if (predict.size() != actual.size()) {
        cerr << "The predictions and actual output size don't match";
    }

    double err = 0.0;
    for (int i = 0; i < predict.size(); i++) {
        double diff = predict[i] - actual[i];
        err += (diff*diff);
    }

    return err / predict.size();
}

//Opens the given file name and pulls the data
//and stores the data in the correct vectors
void pullData(string fileName) {
    //Opening the file
    ifstream File;
    File.open(fileName);

    if (!File.is_open()) {
        cerr << "Failed to open file." << endl;
    }

    //Getting the header
    string header;
    getline(File, header);
    stringstream ss(header);

    //Adding the header names to a list
    //add getting the columns
    string value;

    int columns = 0;

    while (getline(ss, value, ',')) {
        headerNames.push_back(value);
        columns++;
    }

    string line;

    //Reads each line of data and puts into the vectors
    while(getline(File, line)) {
        stringstream lineStream(line);
        vector<double> rowVal;
        string cell;

        while (getline(lineStream, cell, ',')) {
            rowVal.push_back(stod(cell));
        }

        if (rowVal.size() == columns) {
            result.push_back(rowVal.back());
            rowVal.pop_back();
            //Adding a basis
            rowVal.insert(rowVal.begin(), 1.0);

            features.push_back(rowVal);
        }
    }  
}

//Transposes the features matrix
vector<vector<double>> transposeMatrix(vector<vector<double>> matrix) {
    if (matrix.empty()) {
        cerr << "Empty values";
    }

    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<double>> transposed(cols, vector<double>(rows));    

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}
