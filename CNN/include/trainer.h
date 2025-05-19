#ifndef TRAINER_H
#define TRAINER_H

#include "MNIST_loader.h"
#include "cnn_layers.h"
#include "fc_layer.h"
#include <vector>

class Trainer {
public:
    Trainer(vector<vector<float>>& weights,
            vector<float>& bias,
            vector<vector<vector<float>>> kernels,
            float learning_rate = 0.01f);
    void train(MNIST_loader& data, int epochs, int num_samples);

private:
    vector<vector<float>>& fc_weights;
    vector<float>& fc_bias;
    const vector<vector<vector<float>>> kernels;
    const float learning_rate;
};

#endif