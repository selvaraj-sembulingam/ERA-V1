# Grad-CAM



## Folder Structure
```
└── README.md
└── src/
    └── models
        └── resnet.py
    └── data_setup.py
    └── utils.py
    └── engine.py
└── results/
    └── ResNet18.pth
    └── incorrect_images.png
    └── loss_accuracy_plot.png
    └── lr_finder_plot.png
└── train.py
└── S11.ipynb
```

## How to Run the code
Clone the repo and run
Change your current directory to S9
```
python train.py
```

## OneCycle LR

```
from torch_lr_finder import LRFinder

lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state
```

<Insert Image>

## Model Summary

```
```

## Incorrect Classified Images




## Loss Accuracy Plot





## Key Achievements


