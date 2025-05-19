#!/bin/bash
g++ -std=c++11 -o cnn \
    main.cpp \
    src/trainer.cpp \
    src/tester.cpp \
    src/model_io.cpp \
    src/MNIST_loader.cpp \
    src/cnn_layers.cpp \
    src/fc_layer.cpp

./cnn "$@"
