# MLP Housing Price Prediction in C++ (No ML Libraries)

This is an educational project that implements a simple Multilayer Perceptron (MLP) from scratch in C++ to predict housing prices using a CSV dataset. No external machine learning libraries are used.

**Note:** This project was developed as part of a learning exercise. ChatGPT was involved in the design, structure, and implementation of the codebase.

---

## 🚀 Features

- **DataLoader**  
  - CSV parsing with support for numeric and boolean (`True`/`False`) fields  
  - Z-score normalization  
  - Random shuffling and train/test split  
- **MLP Architecture**  
  - Fully connected layers with customizable sizes  
  - ReLU activations in hidden layers  
  - Mean Squared Error loss  
  - Gradient descent training  
- **Evaluation Metrics**  
  - Root Mean Squared Error (RMSE)  
  - “Accuracy” defined as fraction of predictions within ±10% of true price  

---

## 🧠 Model Architecture

**Input (n features)**
  → Dense Layer (64 units) + ReLU
  → Dense Layer (32 units) + ReLU
  → Dense Layer (1 unit) → Output

You can adjust the architecture vector in `main.cpp`:
```cpp
// Example:
// vector<int> arch = { n_features, 64, 32, 1 };
```

## 🛠 How to Build and Run

**Requirements**
  - A C++17-compatible compiler (e.g. g++, clang++)
  - Terminal (Linux/macOS)

**Build**
  - From the project root:
    ```
    g++ -std=c++17 -o main main.cpp src/data_loader.cpp src/mlp.cpp src/Layer.cpp
    ```

## Defaults

**By default, the program will:**
  - Load and normalize data/housing_prices.csv
  - Split into 80% train / 20% test
  - Train for 100 epochs at learning rate 0.01
  - Report training loss each epoch
  - Evaluate on the test set, printing:
  - Test RMSE
  - Accuracy (within ±10%)

## Example Output
```
Train samples: 1168, Test samples: 292
Each input has 13 features.
Epoch 1 / 100 – Loss: 0.543217
Epoch 2 / 100 – Loss: 0.412905
…
Epoch 100 / 100 – Loss: 0.128374

Test RMSE: 0.198542
Accuracy (within ±10%): 83.56%
```

## Project Structure
```
├── data/
│   ├── data_description.txt   # Description of each column in the CSV
│   └── housing_prices.csv     # Raw housing dataset
├── include/                   # Header files
│   ├── data_loader.h
│   ├── layer.h
│   └── mlp.h
├── src/                       # Implementation files
│   ├── data_loader.cpp
│   ├── layer.cpp
│   └── mlp.cpp
├── main.cpp                   # Entry point, configures architecture & runs training/testing
└── README.md
```

## 🙏 Acknowledgments

* Data: [LISETTE](https://www.kaggle.com/datasets/lespin/house-prices-dataset)
* Data Processed: [AHMED NASHAT MAHMOUD](https://www.kaggle.com/code/ahmednashatmahmoud/house-price-preprocessing-task)
* Design and code assistance provided with the help of [ChatGPT](https://openai.com/chatgpt)
  * Creating this README.md and with the project

