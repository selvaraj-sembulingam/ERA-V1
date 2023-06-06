# Backpropagation

Backpropagation is a widely used algorithm for training feedforward neural networks. It computes the gradient of the loss function with respect to the network weights. It is very efficient, rather than naively directly computing the gradient concerning each weight. This efficiency makes it possible to use gradient methods to train multi-layer networks and update weights to minimize loss; variants such as gradient descent or stochastic gradient descent are often used.

The backpropagation algorithm works by computing the gradient of the loss function with respect to each weight via the chain rule, computing the gradient layer by layer, and iterating backward from the last layer to avoid redundant computation of intermediate terms in the chain rule.

Neural networks use supervised learning to generate output vectors from input vectors that the network operates on. It Compares generated output to the desired output and generates an error report if the result does not match the generated output vector. Then it adjusts the weights according to the bug report to get your desired output.

## Neural Network
![simple_perceptron_model-1](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/6b452b12-6830-401a-bcfe-9567cd8d0711)

## Forward pass
First we need to calculate the loss. For that we will use the input and following equations to forward pass the network.
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/372157ce-2362-49d1-87c4-5ec541bd76a6)

## Calculating gradients wrt w5
The following equations are for calculating the gradients with respect to the weight w5
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/e9fdc409-7f72-44ef-8567-a447167bd861)

## Calculating gradients in layer 2
Based on the above equations we will write equations for w5, w6, w7, w8 at layer 2
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/34f7640a-3165-4fab-9eb3-8eb39daa4dec)

## Calculating gradients intermediate step for layer 1
Now for calculating the gradients for layer 1, we need the gradients wrt h1 and h2
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/634cd11c-0802-4e65-a6e9-cffac5b306d0)

## Calculating gradients in layer 1
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/2b28aee4-433c-4cb1-aebf-2902d2181b1a)

## Calculating gradients in layer 1
Using the above equation we can calculate the gradients for layer 1
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/c075c84d-d807-4a56-bf8d-48f3dec2f1a3)

Finally we will update the weights based on the calculated gradients. Based on the new weights, we will calculate the losses and do the above steps iteratively.

## Results for Sample Data (Refer Excel Sheet-1)
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/824d8544-1ef9-4b51-b9f8-e9c8b21c8319)

### Variation of Lossess wrt Learning Rate (Refer Excel Sheet-2)
We can clearly see that as the Learning Rate (in the range from 0.1 to 2) increases, the loss reduces at a faster rate.
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/2f5c34f1-4853-4c4a-b310-a2027c24926f)

