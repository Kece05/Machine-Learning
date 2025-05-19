#include "../include/model_io.h"
#include <iostream>
#include <fstream>

using namespace std;

//Saves the model in a binary file
bool model_IO::save_model(const string& fileName,
                            const vector<vector<float>>& weights,
                            const vector<float>& bias) {
    ofstream out(fileName, ios::binary);
    if (!out) {
        cerr << "Failed to save model";
        return false;
    }

    //Gets the dimensions
    size_t rows = weights.size();
    size_t cols = weights[0].size();

    //Writes the dimensions
    out.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));

    //Writes the weights for each row
    for (const vector<float> row : weights) {
        out.write(reinterpret_cast<const char*>(row.data()), cols * sizeof(float));
    }

    //Writes the bias
    out.write(reinterpret_cast<const char*>(bias.data()), rows * sizeof(float));

    out.close();
    cout << "----------Saved Model------------" << endl;
    return true;
}

//Loads the model in from a binary file
bool model_IO::load_model(const string& fileName,
                            vector<vector<float>>& weights,
                            vector<float>& bias) {
    ifstream in(fileName, ios::binary);

    if (!in) {
        cerr << "Error loading in model";
        return false;
    }

    //Loads the dimensions
    size_t rows;
    size_t cols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&cols), sizeof(size_t));

    weights.resize(rows, vector<float>(cols));
    bias.resize(rows);

    //Reads the weights for each row
    for (vector<float>& row : weights) {
        in.read(reinterpret_cast<char*>(row.data()), cols * sizeof(float));
    }

    //Updates the bias for each row
    in.read(reinterpret_cast<char*>(bias.data()), rows * sizeof(float));

    in.close();
    cout << "----------Loaded Model-----------" << endl;
    return true;
}
