# MNIST CNN in C++ (No ML Libraries)

This is an educational project that implements a simple Convolutional Neural Network (CNN) from scratch in C++ to classify handwritten digits using the MNIST dataset. No external machine learning libraries are used.

**Note:** This project was developed as part of a learning exercise. ChatGPT was involved in the design, structure, and implementation of the codebase.

## 🚀 Features

* CNN architecture with hardcoded 3x3 filters
* ReLU activation
* Max pooling (2x2)
* Flattening for dense layer input
* Fully connected (dense) layer with softmax activation
* Command-line support for training/testing configuration
* Model saving and loading in binary format

## 🧠 Model Architecture

```
Input (28x28 image)
→ Convolution (3 filters)
→ ReLU activation
→ Max Pooling (2x2)
→ Flatten
→ Fully Connected Layer (10 outputs)
→ Softmax
```

## 🛠 How to Build and Run

### Requirements

* A C++11+ compiler
* Terminal (Linux/macOS)
* MNIST `.idx` files placed under `src/data/MNIST/`

### Build and Train
```bash
./run.sh train -e 30 -n 10000 -s FirstModel
```

* `-e`: Number of training epochs (default: 10)
* `-n`: Number of training samples (default: 2500)
* `-s`: Name for saving the trained model (default: cnn_fc_model.bin)

### Example output
```bash
./run.sh train -e 10 -n 10000 -s FirstModel
Epoch 1 | Accuracy: 83.89% | Loss: 0.545961
Epoch 2 | Accuracy: 90.07% | Loss: 0.345446
Epoch 3 | Accuracy: 91.5% | Loss: 0.295429
Epoch 4 | Accuracy: 92.52% | Loss: 0.2648
Epoch 5 | Accuracy: 93.21% | Loss: 0.241497
Epoch 6 | Accuracy: 93.68% | Loss: 0.223476
Epoch 7 | Accuracy: 93.99% | Loss: 0.209178
Epoch 8 | Accuracy: 94.31% | Loss: 0.197072
Epoch 9 | Accuracy: 94.6% | Loss: 0.186472
Epoch 10 | Accuracy: 94.9% | Loss: 0.176939
```

### Test the Model
```bash
./run.sh test -s FirstModel
```

### Example output
```bash
./run.sh test -s FirstModel
----------Loaded Model-----------
Test Accuracy: 90.43%
```

## 📁 Project Structure

```
.
├── main.cpp
├── run.sh
├── include/           # Header files
├── src/               # Implementation files
│   ├── data/MNIST/    # Place MNIST .idx files here
├── models/            # Saved model binaries
```

## 💾 Model Saving

Trained models are saved under the `models/` directory with `.bin` extension and can be reloaded using the `-s` flag during testing.

## 🙏 Acknowledgments

* Data: [Yann LeCun's MNIST dataset](http://yann.lecun.com/exdb/mnist/)
* Design and code assistance provided with the help of [ChatGPT](https://openai.com/chatgpt)

## 📜 License

MIT License — for learning, experimentation, and educational purposes.
