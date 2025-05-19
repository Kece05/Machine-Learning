#ifndef MNIST_loader_H
#define MNIST_loader_H

#include <vector>
#include <string>

using namespace std;

class MNIST_loader {
public:
    vector<vector<unsigned char>> train_images;
    vector<unsigned char> train_labels;
    vector<vector<unsigned char>> test_images;
    vector<unsigned char> test_labels;

    void load(string train_image_path,
              string train_label_path,
              string test_image_path,
              string test_label_path);

private:
    void load_images(string& path, vector<vector<unsigned char>>& images);
    void load_labels(string& path, vector<unsigned char>& labels);
    int reverse_Int(int i);
};

#endif