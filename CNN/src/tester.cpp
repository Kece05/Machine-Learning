#include "../include/tester.h"
#include "../include/MNIST_loader.h"
#include "../include/cnn_layers.h"
#include "../include/fc_layer.h"
#include <vector>
#include <iostream>

using namespace std;

CNN_layers cnn;
FC_layer fc_layer;

//Constructor assigns the variable to the memory location
Tester::Tester(vector<vector<float>>& weights,
            vector<float>& bias,
            vector<vector<vector<float>>> kernels) :
    fc_weights(weights),
    fc_bias(bias),
    kernels(kernels) {}

float Tester::test(MNIST_loader& data, int num_samples) {
    int correct = 0; 

    for (int i = 0; i < num_samples; ++i) {
        //Loads the image and corresponding label, normalizes the image
        vector<vector<float>> image = cnn.normalize_image(data.test_images[i]);
        int label = data.test_labels[i];

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
    }

    //Returning the resultss
    float accuracy = static_cast<float>(correct) / num_samples;
    cout << "Test Accuracy: " << (accuracy * 100.0f) << "%\n";
    return accuracy;
}