#include "../include/data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
#include <algorithm>
#include <random>
#include <cmath>

using namespace std;

//This is the constructor that assigns the inputted string the filename var
dataLoader::dataLoader(const string& fileName) : filename(fileName) {}

//Goes through the dataset and normalizes the data and then returns two
//sets, the data and the price
void dataLoader::load_data() {
    ifstream file(filename);
    string line;

    //Will skip header
    getline(file, line);

    //Will loop through the csv file and add the data to correct place
    while (getline(file, line)) {
        stringstream ss(line);
        string curr_cell;
        vector<double> row;
        double price;

        while (getline(ss, curr_cell, ',')) {
            //Some of the data is T/F
            if (curr_cell == "True") {
                row.push_back(1.0);
            } else if (curr_cell == "False") {
                row.push_back(0.0);
            } else {
                try {
                    row.push_back(stod(curr_cell));
                } catch (...) {
                    cerr << "Error parsing value in row: " << line << "\n";
                    row.clear();
                    break;
                }
            }
        }

        if (!row.empty()) {
            //Getting the price
            price = row.back();
            row.pop_back();

            //Addind to the data 
            data.push_back(row);
            result.push_back(price);
        }
    }

    normalize(data);
    shuffle_data();
    split_data(0.8);

}

//This applies the z-score normalization to the data
//to help the training process smoothly
void dataLoader::normalize(vector<vector<double>>& data) {
    if (data.empty()) {
        return;
    }

    size_t num_features = data[0].size();
    means.assign(num_features, 0.0);
    std_devs.assign(num_features, 0.0);
    size_t n = data.size();

    //Computing mean
    for (const vector<double>& row : data) {
        for (size_t i = 0; i < num_features; ++i) {
            means[i] += row[i];
        }
    }

    for (size_t i = 0; i < num_features; ++i) {
        means[i] /= n;
    }

    //Computing the suqared standard devivation
    for (const vector<double>& row : data) {
        for (size_t i = 0; i < num_features; ++i) {
            double diff = row[i] - means[i];
            std_devs[i] += diff * diff;
        }
    }
    
    //Finalize standard deviations
    for (size_t i = 0; i < num_features; ++i) {
        std_devs[i] = sqrt(std_devs[i] / n);
        if (std_devs[i] == 0.0)
            std_devs[i] = 1.0;
    }

    //Applying the z-score normalization
    for (vector<double>& row : data) {
        for (size_t i = 0; i < num_features; ++i) {
            row[i] = (row[i] - means[i]) / std_devs[i];
        }
    }
}

//Shuffling the dataset 
void dataLoader::shuffle_data() {
    vector<size_t> indices(data.size());
    for (size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;

    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);

    vector<vector<double>> data_shuffled;
    vector<double> result_shuffled;

    for (size_t i : indices) {
        data_shuffled.push_back(data[i]);
        result_shuffled.push_back(result[i]);
    }

    data = std::move(data_shuffled);
    result = std::move(result_shuffled);
}

//Splits the data into two categories: training and testing 
void dataLoader::split_data(double train_ratio) {
    size_t total = data.size();

    //Getting the size of the train set by total * training ratio
    size_t train_size = static_cast<size_t>(total * train_ratio);

    //Limiting the two data sets to the size
    data_train.assign(data.begin(), data.begin() + train_size);
    result_train.assign(result.begin(), result.begin() + train_size);

    data_test.assign(data.begin() + train_size, data.end());
    result_test.assign(result.begin() + train_size, result.end());
}

//Getter for the training data
pair<vector<vector<double>>, vector<double>> dataLoader::get_train() const {
    return {data_train, result_train};
}

//Getter for the test data
pair<vector<vector<double>>, vector<double>> dataLoader::get_test() const {
    return {data_test, result_test};
}