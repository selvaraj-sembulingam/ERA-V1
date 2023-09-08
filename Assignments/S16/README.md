# Training Transformers Efficiently

The Transformer model has gained significant popularity in the field of natural language processing (NLP) due to its outstanding performance on various tasks. However, training large Transformer models can be computationally intensive and time-consuming. To address these challenges, this repository explores several advanced training techniques to make the training process more efficient and effective.

<img src="https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/362f1888-8953-4abb-a2a8-1d9bcdebae79" width=50% height=50%>

## Dynamic Padding
Dynamic padding is a technique that optimizes the padding of input sequences during training. Instead of padding all sequences to the maximum length in a batch, dynamic padding pads each batch to the length of the longest sequence in that batch. This reduces the amount of unnecessary computation and speeds up training.

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/6404e10e-26bb-455a-a42d-75115f75cbf6)


## Automatic Mixed Precision
Automatic Mixed Precision (AMP) is a method that combines 16-bit and 32-bit floating-point arithmetic to accelerate training while maintaining model accuracy. This technique takes advantage of hardware acceleration for faster training without sacrificing model quality.

## Parameter Sharing
Parameter sharing is a technique where you share model parameters across layers or model components. This can lead to a reduction in the number of parameters in the model, making it more memory-efficient and potentially improving generalization.

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/12931625-b0df-4191-af4a-c620c79395b3)


## One Cycle Policy
The One Cycle Policy is a learning rate schedule that varies the learning rate during training to achieve faster convergence and better final performance. This policy involves gradually increasing and then decreasing the learning rate during training.

## Achievements
Reached a loss of 1.787 in 16 epochs for en-fr dataset

