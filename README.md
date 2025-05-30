# Machine Learning in C++ (From Scratch)

This repository contains educational implementations of machine learning algorithms built from scratch in C++. The goal is to explore and understand the inner workings of models such as linear regression and convolutional neural networks (CNNs) without relying on external machine learning libraries.

> ⚠️ **Disclaimer**: This project was built for educational purposes and assisted by ChatGPT in both structure and implementation guidance.

---

## 📁 Contents

### 1. Least Squares Regression

This project performs linear regression on a dataset provided in a CSV file. The program reads features such as hours studied, attendance rate, and homework/quiz scores to predict a final exam score using the Least Squares Method.

- 📄 Input: CSV file with labeled training data.
- ⚙️ Output: Trained weights and bias for prediction.
- 💡 Functionality:
  - Calculates weights using the formula: **θ = (QᵀQ)⁻¹ Qᵀb**
  - Prints Mean Square Error
  - Allows user to make final score predictions with custom inputs


---

### 2. Convolutional Neural Network (CNN)

This project is a basic Convolutional Neural Network implemented from scratch to classify MNIST digits.

- 🔢 Layers: Convolution → ReLU → Max Pool → Flatten → Fully Connected
- 💾 Model can be trained and saved to file
- 🧪 Accuracy output per epoch

---

### 3. Multi-Layer Perceptron (MLP)
This project implements a simple Multilayer Perceptron (MLP) from scratch in C++ to predict housing prices using a CSV dataset.  

- 📄 Input: `housing_prices.csv` with 13 features per record  
- ⚙️ Output: Trained MLP model saved in binary format  
- 💡 Functionality:  
  - Z-score normalization of features  
  - Train/test split (80/20) with random shuffling  
  - Configurable architecture (e.g. `{13, 64, 32, 1}`)  
  - ReLU activations and MSE loss  
  - Reports training loss per epoch  
  - Evaluates test set: RMSE and “accuracy” (predictions within ±10% of true)  

---

### 🤝 Credits
This project was created by Kece05 with implementation structure and sample code support provided by ChatGPT. Designed for educational purposes to deepen understanding of machine learning foundations.
