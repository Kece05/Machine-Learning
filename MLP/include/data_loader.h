#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <string>
#include <utility>

using namespace std;

class dataLoader {
public:
    dataLoader(const string& fileName);
    void load_data();
    
    pair<vector<vector<double>>, vector<double>> get_train() const;
    pair<vector<vector<double>>, vector<double>> get_test() const;

private:
    vector<double> means, std_devs;
    string filename;

    vector<vector<double>> data;
    vector<double> result;

    vector<vector<double>> data_train, data_test;
    vector<double> result_train, result_test;

    void normalize(vector<vector<double>>& data);
    void shuffle_data();
    void split_data(double train_ratio);
};

#endif