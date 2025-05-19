#ifndef CNN_LAYERS_H
#define CNN_LAYERS_H

#include <vector>

using namespace std;

class CNN_layers {
public:
    vector<vector<float>> normalize_image(const vector<unsigned char>& image_1d);
    vector<vector<float>> convolve2D(const vector<vector<float>>&, const vector<vector<float>>&);
    void apply_relu(vector<vector<vector<float>>>& feature_maps);
    vector<vector<float>> max_pool2x2(const vector<vector<float>>&);
    vector<float> flatten(const vector<vector<vector<float>>>& maps);

private:
    float relu(float x);
};

#endif
