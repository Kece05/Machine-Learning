#ifndef MLP_H
#define MLP_H

#include <vector>
#include <string>
#include "layer.h"

using namespace std;

class MLP {
public:
    MLP(const vector<int>& arch);
    void train(const vector<vector<double>>& weights,
                const vector<double>& results,
                int epochs, double learning_rate);
    double predict(const vector<double>& x);
    vector<double> predict_batch(const vector<vector<double>>& X);

private:
    vector<Layer> layers;

    static double loss(const vector<double>& pred, const vector<double>& real);
    static vector<double> loss_dervivate(const vector<double>& pred, 
                                        const vector<double>& real);
};

#endif