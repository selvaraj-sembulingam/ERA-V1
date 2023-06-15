# Convolutional Neural Network for MNIST

## Model 1

### Target
* Get the basic skeleton right. Try and avoid changing this skeleton as much as possible. 
* No fancy stuff
### Results:
* Parameters: 139,808
* Best Train Accuracy: 99.66
* Best Test Accuracy: 99.08
### Analysis:
* The model is still large, but working. 
* Over-fitting is seen

## Model 2

### Target
* Make Model lighter
### Results:
* Parameters: 6,514
* Best Train Accuracy: 99.08
* Best Test Accuracy: 98.86
### Analysis:
* Good model!
* No Overfitting. Model is capable if pushed further

## Model 3

### Target
* Add Batch-norm to increase model efficiency.
### Results:
* Parameters: 6,646
* Best Train Accuracy: 99.66
* Best Test Accuracy: 99.30
### Analysis:
* Started to see over-fitting now. 
* Even if the model is pushed further, it won't be able to get to 99.4

## Model 4

### Target
* Add Regularization, Dropout
### Results:
* Parameters: 6,646
* Best Train Accuracy: 99.56
* Best Test Accuracy: 99.38
### Analysis:
* Regularization working. The difference between train and test accuracy is reduced.
* Seeing image samples, we can add slight rotation. 

## Model 5

### Target
* Add rotation, guess is that 5-7 degrees should be sufficient. 
### Results:
* Parameters: 6,646
* Best Train Accuracy: 99.26
* Best Test Accuracy: 99.35
### Analysis:
* The model is under-fitting now. This is fine, as train data is made harder. 

## Model 6

### Target
* Add LR Scheduler
### Results:
* Parameters: 6,646
* Best Train Accuracy: 99.43
* Best Test Accuracy: 99.47
### Analysis:
* Tried to make it effective by using ReduceLROnPlateau
* Achieved test accuracy more thatn 99.4


## Receptive Field of the Models
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/ba2e8219-45b3-4746-8d5d-dfb031177e2e)
