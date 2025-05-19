#include "../include/MNIST_loader.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;


vector<vector<unsigned char>> train_images;
vector<unsigned char> train_labels;
vector<vector<unsigned char>> test_images;
vector<unsigned char> test_labels;

void MNIST_loader::load(string train_image_path,
                string train_label_path,
                string test_image_path,
                string test_label_path) {
                    load_images(train_image_path, train_images);
                    load_images(test_image_path, test_images);
                    load_labels(train_label_path, train_labels);
                    load_labels(test_label_path, test_labels);
}

void MNIST_loader::load_images(string& path, vector<vector<unsigned char>>& images) {
    ifstream imageFile(path, ios::binary);
    if (!imageFile.is_open()) {
        cerr << "Failed to open file: " << path << std::endl;
        exit(1);
    }
    int magic = 0;
    int numImage = 0;
    int numRows = 0;
    int numCols = 0;

    imageFile.read(reinterpret_cast<char*>(&magic), 4);
    magic = reverse_Int(magic);

    imageFile.read(reinterpret_cast<char*>(&numImage), 4);
    numImage = reverse_Int(numImage);

    imageFile.read(reinterpret_cast<char*>(&numRows), 4);
    numRows = reverse_Int(numRows);

    imageFile.read(reinterpret_cast<char*>(&numCols), 4);
    numCols = reverse_Int(numCols);

    images.resize(numImage, vector<unsigned char>(numRows * numCols));
    for (int i = 0; i < numImage; ++i) {
        imageFile.read(reinterpret_cast<char*>(images[i].data()), numRows * numCols);
    }

    imageFile.close();
}

void MNIST_loader::load_labels(string& path, vector<unsigned char>& labels) {
    ifstream imageFile(path, ios::binary);
    if (!(imageFile.is_open())) {
        cerr << "File cannot open";
    }

    int magic = 0;
    int numLabels = 0;

    imageFile.read(reinterpret_cast<char*>(&magic), 4);
    magic = reverse_Int(magic);

    imageFile.read(reinterpret_cast<char*>(&numLabels), 4);
    numLabels = reverse_Int(numLabels);

    labels.resize(numLabels);
    imageFile.read(reinterpret_cast<char*>(labels.data()), numLabels);
    imageFile.close();
}

int MNIST_loader::reverse_Int(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) | ((int)c2 << 16) | ((int)c3 << 8) | c4;
}
