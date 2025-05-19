#ifndef MODEL_IO_H
#define MODEL_IO_H

#include <vector>
#include <string>

using namespace std;

class model_IO {
public:
    static bool save_model(const string& fileName,
                            const vector<vector<float>>& weights,
                            const vector<float>& bias);
    static bool load_model(const string& fileName,
                            vector<vector<float>>& weights,
                            vector<float>& bias);

};

#endif