#include "../include/cnn_layers.h"

using namespace std;

//This converts the images from 1d to 2d images(28x28) 
vector<vector<float>> CNN_layers::normalize_image(const vector<unsigned char>& image_1d) {
    int rows = 28;
    int cols = 28;

    vector<vector<float>> image_2d(rows, vector<float>(cols));

    //Pulls the pixel and corresponding the 2d image and assigns a value 0-1
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            image_2d[i][j] = static_cast<float>(image_1d[i * cols + j]) / 255.0f;
        }
    }

    return image_2d;
}

//This applies filters on the images and given the kernel 
//Applied to each pixel - the output size is input - kernal + 1
vector<vector<float>> CNN_layers::convolve2D(const vector<vector<float>>& input,
                                const vector<vector<float>>& kernel) {
    int input_rows = input.size();
    int input_cols = input[0].size();
    int k_size = kernel.size();
    int output_size = input_rows - k_size + 1;

    vector<vector<float>> output(output_size, vector<float>(output_size, 0.0f));

    //Sliding the kernel(ki and kj) across the image and doing the dot product
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float sum = 0.0f;
            for (int ki = 0; ki < k_size; ++ki) {
                for (int kj = 0; kj < k_size; ++kj) {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }

    return output;
}

//If less than 0 then makes the pixel 0 or keeps the value
float CNN_layers::relu(float x) {
    return max(0.0f, x);
}

//This applies the relu function to the image
void CNN_layers::apply_relu(vector<vector<vector<float>>>& feature_maps) {
    for (vector<vector<float>>& feature_map : feature_maps) {
        for (vector<float>& row : feature_map) {
            for (auto& val : row) {
                val = relu(val);
            }
        }
    }
}

//This downsizes the image by half while keeping prominent features
vector<vector<float>> CNN_layers::max_pool2x2(const vector<vector<float>>& input) {
    int out_rows = input.size() / 2;
    int out_cols = input[0].size() / 2;
    vector<vector<float>> pooled(out_rows, vector<float>(out_cols));

    //Finds the max pixel value and that is choosen for the pool image
    for (int i = 0; i < out_rows; ++i) {
        for (int j = 0; j < out_cols; ++j) {
            float max_val = input[i * 2][j * 2];
            for (int di = 0; di < 2; ++di) {
                for (int dj = 0; dj < 2; ++dj) {
                    max_val = max(max_val, input[i * 2 + di][j * 2 + dj]);
                }
            }
            pooled[i][j] = max_val;
        }
    }

    return pooled;
}

//Flattens each map into a 1d vector
vector<float> CNN_layers::flatten(const vector<vector<vector<float>>>& maps) {
    vector<float> flat;
    for (const vector<vector<float>>& fmap : maps)
        for (const vector<float>& row : fmap)
            for (float val : row)
                flat.push_back(val);
    return flat;
}