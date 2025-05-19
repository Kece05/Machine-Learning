#include "../include/trainer.h"
#include "../include/cnn_layers.h"
#include "../include/fc_layer.h"
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace std;

//Constructor assigns the variable to the memory location
Trainer::Trainer(vector<vector<float>>& weights,
            vector<float>& bias,
            vector<vector<vector<float>>> kernels,
            float learning_rate) :
    fc_weights(weights),
    fc_bias(bias),
    kernels(kernels),
    learning_rate(learning_rate) {}

//This trains the model through repetition of the training data
void Trainer::train(MNIST_loader& data, int epochs, int num_samples) {
    CNN_layers cnn;
    FC_layer fc_layer;
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        int correct = 0;
        float total_loss = 0.0f;

        for (int i = 0; i < num_samples; ++i) {
            //Loads the image and corresponding label, normalizes the image
            vector<vector<float>> image = cnn.normalize_image(data.train_images[i]);
            int label = data.train_labels[i];

            vector<vector<vector<float>>> feature_maps;

            //Applies the features to the image
            for (const vector<vector<float>>& kernel : kernels) {
                feature_maps.push_back(cnn.convolve2D(image, kernel));
            }

            cnn.apply_relu(feature_maps);

            //Resizes the image while keeping prominant features
            vector<vector<vector<float>>> pooled_maps;
            for(const vector<vector<float>>& feature_map : feature_maps) {
                pooled_maps.push_back(cnn.max_pool2x2(feature_map));
            }

            //Flattens the image for the fc layer
            vector<float> flat_input = cnn.flatten(pooled_maps);

            //Fully connected forward pass
            vector<float> logits(10, 0.0f);

            for (int j = 0; j < 10; ++j) {
                for (size_t k = 0; k < flat_input.size(); ++k) {
                    logits[j] += flat_input[k] * fc_weights[j][k];
                }
                logits[j] += fc_bias[j];
            }
            
            //Getting and comparing the prediction of the model to the actual label
            vector<float> probs = fc_layer.softmax(logits);
            int prediction = distance(probs.begin(), max_element(probs.begin(), probs.end()));
            if (prediction == label) {
                correct++;
            }

            //Calculating the loss
            total_loss += fc_layer.cross_entropy_loss(probs, label);

            //Computing the gradient and updating the weight accordingly
            vector<float> grad = fc_layer.output_gradient(probs, label);
            fc_layer.update_fc_layer(fc_weights, fc_bias, flat_input, grad, learning_rate);
        }

        //Print out results
        cout << "Epoch " << (epoch + 1)
                << " | Accuracy: " << (100.0 * correct / num_samples) << "%"
                << " | Loss: " << (total_loss / num_samples) << "\n";
    }
}