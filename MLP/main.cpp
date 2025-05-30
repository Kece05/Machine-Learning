#include "include/data_loader.h"
#include "include/mlp.h"
#include <iostream>
#include <cmath>

using namespace std;

int main() {
    dataLoader set("/Path to file/MLP/data/housing_prices.csv");

    set.load_data();

    //Loadint two datasets - train and test
    auto [X_train, y_train] = set.get_train();
    auto [X_test, y_test] = set.get_test();

    cout << "Train samples: " << X_train.size() << ", Test samples: " << X_test.size() << "\n";
    cout << "Each input has " << X_train[0].size() << " features.\n";

    //Creating the archetecture of the model and loading it in
    vector<int> arch = {
        static_cast<int>(X_train[0].size()), 
        64, 32, 1};
        
    MLP model(arch);

    //Traing model
    model.train(X_train, y_train, 100, 0.01);

    //Testing model
    vector<double> preds = model.predict_batch(X_test);
    double mse = 0.0;
    int correct = 0;
    double tol = 0.10;

    for (size_t i = 0; i < preds.size(); ++i) {
        double diff = preds[i] - y_test[i];
        mse += diff * diff;

        if (fabs(diff) / max(y_test[i], 1e-8) < tol) {
            ++correct;
        }
    }

    mse /= static_cast<int>(preds.size());
    double rmse = sqrt(mse);
    double accuracy = 100.0 * correct / static_cast<double>(preds.size());
    
    cout << "Test RMSE: " << rmse << endl;
    cout << "Accuracy (within ±" << (tol*100) << "%): " << accuracy << "%" << endl;

    return 0;
}
