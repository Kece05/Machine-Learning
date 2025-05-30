#include "../include/layer.h"
#include <cmath>
#include <random>
#include <vector>
#include <iostream>

using namespace std;

Layer::Layer(int input_size, int output_size) 
    : input_size(input_size), output_size(output_size) {

    double limit = sqrt( 2.0 / input_size);
        
    //Initializing neuron sizes for the layer
    weights.resize(output_size, vector<double>(input_size));
    bais.assign(output_size, 0.0);

    //Assiging random values for each weight
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights[i][j] = random_weight(limit);
        }
    }

} 

//Creates random set of doubles from negative infinity to infinity
//For inital weights
double Layer::random_weight(double limit) {
    static mt19937 gen(random_device{}());
    uniform_real_distribution<double> dist(-limit, limit);
    return dist(gen);
}

//This computes the sum of each neuron weighted input and activates
vector<double> Layer::forward(const vector<double>& input) {
    inputs = input;
    outputs.assign(output_size, 0.0);
    
    //Calculates the weight of each neurons input
    //And the uses ReLU
    for (int i = 0; i < output_size; ++i) {
        double sum = bais[i];
        for (int j = 0; j < input_size; ++j) {
            sum += weights[i][j] * input[j];
        }

        outputs[i] = activate(sum);
    }

    return outputs;
}

//Updates the weights and biases while also computing the gradient for the previous layer
vector<double> Layer::backward(const vector<double>& grad_output, double learning_rate) {
    //Calculating delta
    vector<double> delta(output_size);
    for (int i = 0; i < output_size; ++i) {
        delta[i] = grad_output[i] * activate_derv(outputs[i]);
    }

    //Computes the gradient wrt for inputs used for backwards propogation
    vector<double> grad_input(input_size, 0.0);
    for (int j = 0; j < output_size; ++j) {
        for (int i = 0; i < input_size; ++i) {
            grad_input[i] += weights[j][i] * delta[j];
        }
    }

    //Updating weights and biases using gradient descent
    for (int j = 0; j < output_size; ++j) {
        for (int i = 0; i < input_size; ++i) {
            weights[j][i] -= learning_rate * delta[j] * inputs[i];
        }
        bais[j] -= learning_rate * delta[j];
    }

    return grad_input;  
}