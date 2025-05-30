#ifndef LAYER_H
#define LAYER_H

#include <vector>

using namespace std;

class Layer {
public:
    Layer(int input_size, int output_size);

    vector<double> forward(const vector<double>& input);
    vector<double> backward(const vector<double>& grad_output, double learning_rate);

private:
    int input_size;
    int output_size;

    vector<vector<double>> weights;
    vector<double> bais;

    vector<double> inputs;
    vector<double> outputs;

    double random_weight(double limit);

    //Uses ReLU
    static double activate(double x) {
        return (x > 0) ? x : 0.0;
    }

    static double activate_derv(double x) {
        return (x > 0) ? 1.0 : 0.0;
    }
};

#endif