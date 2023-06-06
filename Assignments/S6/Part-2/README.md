# Convolutional Neural Network for MNIST

## Architecture
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/1f32ee9a-d558-454e-9298-7773b2241f34)

## Description

The architecture is a deep convolutional neural network (CNN) which achieved outstanding performance on MNIST image classification. The key characteristic is its simplicity and uniformity in design, making it easy to understand and replicate.

The  architecture consists of several convolutional layers followed by GAP.

The core building block is the repeated use of 3x3 convolutional layers stacked on top of each other. The basic structure of an architectural block is as follows:

Convolutional Layers: A set of consecutive convolutional layers with a small receptive field (3x3). These layers are responsible for capturing local features and spatial patterns in the input images.

BatchNorm: To normalize the inputs of a layer by subtracting the batch mean and dividing by the batch standard deviation, helping to stabilize and accelerate the training process.

ReLU Activation: A rectified linear unit (ReLU) activation function is applied after each convolutional layer and max pooling layer. ReLU introduces non-linearity to the network, allowing it to model complex relationships in the data.

Max Pooling Layer: A max pooling layer follows the convolutional layers. It performs downsampling by taking the maximum value within a pooling window. Max pooling helps in reducing the spatial dimensions while retaining the most salient features.

Finally a Global Average Pooling and softmax are used to produce predictions
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
       BatchNorm2d-2            [-1, 8, 28, 28]              16
              ReLU-3            [-1, 8, 28, 28]               0
            Conv2d-4            [-1, 8, 28, 28]             584
       BatchNorm2d-5            [-1, 8, 28, 28]              16
              ReLU-6            [-1, 8, 28, 28]               0
            Conv2d-7            [-1, 8, 28, 28]             584
       BatchNorm2d-8            [-1, 8, 28, 28]              16
              ReLU-9            [-1, 8, 28, 28]               0
        MaxPool2d-10            [-1, 8, 14, 14]               0
           Conv2d-11           [-1, 16, 14, 14]           1,168
      BatchNorm2d-12           [-1, 16, 14, 14]              32
             ReLU-13           [-1, 16, 14, 14]               0
           Conv2d-14           [-1, 16, 14, 14]           2,320
      BatchNorm2d-15           [-1, 16, 14, 14]              32
             ReLU-16           [-1, 16, 14, 14]               0
        MaxPool2d-17             [-1, 16, 7, 7]               0
           Conv2d-18             [-1, 32, 7, 7]           4,640
      BatchNorm2d-19             [-1, 32, 7, 7]              64
             ReLU-20             [-1, 32, 7, 7]               0
           Conv2d-21             [-1, 32, 7, 7]           9,248
      BatchNorm2d-22             [-1, 32, 7, 7]              64
             ReLU-23             [-1, 32, 7, 7]               0
        MaxPool2d-24             [-1, 32, 3, 3]               0
           Conv2d-25             [-1, 10, 3, 3]             330
        AvgPool2d-26             [-1, 10, 1, 1]               0
================================================================
Total params: 19,194
Trainable params: 19,194
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 0.07
Estimated Total Size (MB): 0.74
----------------------------------------------------------------
```

## Key Achievements
* 99.4% validation accuracy
* Less than 20k Parameters
* Less than 20 Epochs
