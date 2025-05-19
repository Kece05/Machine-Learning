#ifndef TESTER_H
#define TESTER_H

#include "MNIST_loader.h"
#include "cnn_layers.h"
#include "fc_layer.h"
#include <vector>

using namespace std;

class Tester {
public:
    Tester(vector<vector<float>>& weights,
            vector<float>& bias,
            vector<vector<vector<float>>> kernels);

    float test(MNIST_loader& data, int num_samples);

private:
    const vector<vector<float>>& fc_weights;
    const vector<float>& fc_bias;
    const vector<vector<vector<float>>> kernels;
};

#endif