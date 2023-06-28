# Advanced Convolutions and Data Augmentation

## Folder Structure
```
└── README.md
└── src/
    └── data_setup.py
    └── utils.py
    └── engine.py
    └── model_builder.py
    └── model
└── models/
    └── S9Model1.pth
    └── incorrect_images.png
    └── loss_accuracy_plot.png
└── train.py
└── S9.ipynb
```

## How to Run the code
Clone the repo and run
Change your current directory to S9
```
python train.py
```

## Receptive Field Calculations
| |r_in|n_in|j_in|s|r_out|n_out|j_out| |kernal_size|padding|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|Conv |1|32|1|1|3|32|1| |3|1|
|Conv|3|32|1|1|5|32|1| |3|1|
|Conv (Dilated)|5|32|1|1|9|30|1| |5|1|
|Conv|9|30|1|1|11|30|1| |3|1|
|Conv|11|30|1|1|13|30|1| |3|1|
|Conv (Stride2)|13|30|1|2|15|15|2| |3|1|
|Conv|15|15|2|1|19|15|2| |3|1|
|Conv|19|15|2|1|23|15|2| |3|1|
|Conv (Stride2)|23|15|2|2|27|8|4| |3|1|
|Conv (DWS)|27|8|4|1|35|8|4| |3|1|
|Conv|35|8|4|1|43|8|4| |3|1|
|Conv |43|8|4|1|51|8|4| |3|1|
|GAP|51|8|4|1|79|1|4| |8|0|

## Convolutions

### Normal Convolution

### Strided Convolution (stride = 2)
![1_NrsBkY8ujrGlq83f8FR2wQ](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/c5e1502f-1f8a-4c9e-8f7a-1f924dd690ad)


### Dilated Convolution (dilation=2)
![1_niGh2BkLuAUS2lkctkd3sA](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/a55d83cb-482f-4995-aab6-036f6be55066)


## Training and Testing Results
* Best Train Accuracy: 82.18
* Best Test Accuracy: 86.78

![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/2bba9013-cc06-44d1-9546-c66b6875cb93)


## Incorrect Classified Images
![image](https://github.com/selvaraj-sembulingam/ERA-V1/assets/66372829/53c474ec-95a0-468a-9200-3a6a6aa76324)
