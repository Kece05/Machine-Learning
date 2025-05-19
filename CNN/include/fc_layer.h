#ifndef FC_LAYER_H
#define FC_LAYER_H

#include <vector>

using namespace std;

class FC_layer {
public:
    vector<float> softmax(const vector<float>& input);
    float cross_entropy_loss(const vector<float>& probs, int label);
    vector<float> output_gradient(const vector<float>& probs, int label);

    void update_fc_layer(vector<vector<float>>& weights,
                        vector<float>& bias,
                        const vector<float>& input,
                        const vector<float>& grad,
                        float learning_rate);
};

#endif
