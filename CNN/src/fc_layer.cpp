#include "../include/fc_layer.h"
#include <cmath>
#include <algorithm>

using namespace std;

//This applies the softmax function to get the probility of each
//output
vector<float> FC_layer::softmax(const vector<float>& input) {
    vector<float> output(input.size());
    float sum = 0.0f;

    //Gets the sum of the denimator
    for (float x : input) {
        sum += exp(x);
    }
    //Calculates each outputs softmax
    for (size_t i = 0; i < input.size(); ++i) {
         output[i] = exp(input[i]) / sum;
    }
    return output;
}

//This calculates the cross entropy loss between the predicted label and 
//the correct label
float FC_layer::cross_entropy_loss(const vector<float>& probs, int label) {
    //Using clamp method to avoid crashes like log(0)
    float p = max(1e-9f, min(1.0f - 1e-9f, probs[label]));
    return -log(p);
}

//This calcualates the gradient of the softmax and cross entropy loss to update
//the weights and is used for backpropagation 
vector<float> FC_layer::output_gradient(const vector<float>& probs, int label) {
    vector<float> grad = probs;
    grad[label] -= 1.0f; //The dL/dz formula
    return grad;
}

//Updates the weights of each layer biases with the gradient descent
//which is also adjusted by the learning rate
void FC_layer::update_fc_layer(vector<vector<float>>& weights,
                     vector<float>& bias,
                     const vector<float>& input,
                     const vector<float>& grad,
                     float learning_rate) {
    for (int i = 0; i < 10; ++i) {
        for (size_t j = 0; j < input.size(); ++j) {
            weights[i][j] -= learning_rate * grad[i] * input[j];
        }
        bias[i] -= learning_rate * grad[i];
    }
}
