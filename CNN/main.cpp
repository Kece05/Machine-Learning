#include "include/MNIST_loader.h"
#include "include/cnn_layers.h"
#include "include/fc_layer.h"
#include "include/model_io.h"
#include "include/trainer.h"
#include "include/tester.h"

#include <iostream>
#include <vector>
#include <ctime>
#include <algorithm>
#include <random>

using namespace std;

void shuffleImages(MNIST_loader& images);

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage:\n"
             << "  ./cnn train [-e epochs] [-n num_images] [-s File Name]\n"
             << "  ./cnn test\n";
        return 1;
    }

    srand(static_cast<unsigned>(time(nullptr)));

    //Load data set
    MNIST_loader mnist;
    mnist.load(
        "/Users/kece/Desktop/MacOS/cpp/CNN/src/data/MNIST/train-images.idx3-ubyte",
        "/Users/kece/Desktop/MacOS/cpp/CNN/src/data/MNIST/train-labels.idx1-ubyte",
        "/Users/kece/Desktop/MacOS/cpp/CNN/src/data/MNIST/t10k-images.idx3-ubyte",
        "/Users/kece/Desktop/MacOS/cpp/CNN/src/data/MNIST/t10k-labels.idx1-ubyte");

    //Kernels that will be used in the feature maps
    vector<vector<vector<float>>> kernels = {
        {
            { -1,  0,  1 },
            { -2,  0,  2 },
            { -1,  0,  1 }
        },
        {
            { 0,  1,  0 },
            { 1, -4,  1 },
            { 0,  1,  0 }
        },
        {
            { 1,  1,  1 },
            { 1, -8,  1 },
            { 1,  1,  1 }
        }
    };

    //Initalizing the weights and bias
    int flat_size = 13 * 13 * kernels.size();
    vector<vector<float>> fc_weights(10, vector<float>(flat_size));
    vector<float> fc_bias(10, 0.0f);

    for (vector<float>& row : fc_weights) {
        for (float& weight : row) {
            //Randomly initalize each weight
            weight = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }

    string mode = argv[1];
    string fileName = "cnn_fc_model.bin";

    //Default setting for not given args
    int epochs = 10;
    int num_images = 2500;

    //Looking to see if the are args
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            epochs = stoi(argv[++i]);
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_images = stoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            fileName = argv[++i] + string(".bin");
        }
    }

    if (mode == "train") {
        shuffleImages(mnist);

        //Training model
        Trainer trainer(fc_weights, fc_bias, kernels, 0.01f);
        trainer.train(mnist, epochs, num_images);

        //Saving model
        model_IO::save_model("/Users/kece/Desktop/MacOS/cpp/CNN/models/" + fileName, fc_weights, fc_bias);
    } else if (mode == "test") {
        //Loading model
        model_IO::load_model("/Users/kece/Desktop/MacOS/cpp/CNN/models/" + fileName, fc_weights, fc_bias);

        //Testing model
        Tester tester(fc_weights, fc_bias, kernels);
        tester.test(mnist, mnist.test_images.size());
    } else {
        cerr << "Invalid mode. Use 'train' or 'test'.\n";
        return 1;
    }

    return 0;
}

//Shuffling images so it is randomized
void shuffleImages(MNIST_loader& images) {
    //Creating a random seed
    random_device rd;
    mt19937 gen(rd());

    //Creating a list of indices corresponding to the images
    vector<size_t> indices(images.train_images.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    //Shuffleing new set
    shuffle(indices.begin(), indices.end(), gen);

    //Creating a new set of images and labels 
    vector<vector<unsigned char>> shuffled_train_images;
    vector<unsigned char> shuffled_train_labels;

    //Adding orginial images to the shuffled set
    for (size_t idx : indices) {
        shuffled_train_images.push_back(images.train_images[idx]);
        shuffled_train_labels.push_back(images.train_labels[idx]);
    }

    //Updating the orginial images 
    images.train_images = shuffled_train_images;
    images.train_labels = shuffled_train_labels;
}