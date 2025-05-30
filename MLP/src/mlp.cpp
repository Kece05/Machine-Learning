#include "../include/mlp.h"
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

//Setting up each layer of the algothirm
MLP::MLP(const vector<int>& arch) {
    for (size_t i = 1; i < arch.size(); ++i) {
        //loading in each layer with input - output
        layers.emplace_back(arch[i-1], arch[i]);
    }
}

void MLP::train(const vector<vector<double>>& weights,
                const vector<double>& results,
                int epochs, double learning_rate) {

    size_t n_samples = weights.size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;

        for (int i = 0; i < n_samples; ++i) {
            vector<double> activation = weights[i];

            //Feeding data through each layer - forward propagation
            for (Layer& layer : layers) {
                activation = layer.forward(activation);
            }

            //Gets the predicted value and the actual value
            //to calculate the loss
            vector<double> pred = activation;
            vector<double> actual = { results[i] };

            //Calculating the loss
            epoch_loss += loss(pred, actual);

            //Backward propagation
            //Computing the gradient wrt loss
            vector<double> grad = loss_dervivate(pred, actual);

            //Updating each layer/weight starting 
            for (int l = layers.size() - 1; l >= 0; --l) {
                grad = layers[l].backward(grad, learning_rate);
            }
        }
        epoch_loss /= static_cast<double>(n_samples);
        cout << "Epoch " << epoch+1 << " / " << epochs << " - Loss: " << epoch_loss << endl;
    }
}

//Calculates the mean square error
double MLP::loss(const vector<double>& pred, const vector<double>& real) {
    double sum = 0.0;

    for (size_t i = 0; i < pred.size(); ++i) {
        double delta = (pred[i] - real[i]);

        sum += (delta * delta);
    }

    return (sum / static_cast<double>(pred.size()));
}

//Calculates the dervivate of the mean square error
//which helps to update weights that would make the predictions 
//closer to the actual result
vector<double> MLP::loss_dervivate(const vector<double>& pred, 
                                        const vector<double>& real) {
    vector<double> grad(pred.size());

    for (size_t i = 0; i < pred.size(); ++i) {
        grad[i] = (2 * (pred[i] - real[i])) / static_cast<double>(pred.size());
    }

    return grad;
}

//This passes in the data and feeds it through the layers
//and then gets the value predicted
double MLP::predict(const vector<double>& x) {
    vector<double> activation = x;
    for (Layer& layer : layers) {
        activation = layer.forward(activation);
    }

    return activation.empty() ? 0.0 : activation[0];
}

//This is used to predict multiple samples
vector<double> MLP::predict_batch(const vector<vector<double>>& X) {
    vector<double> preds;
    preds.reserve(X.size());

    for (const vector<double>& x : X) {
        preds.push_back(predict(x));
    }

    return preds;
}