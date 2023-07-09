import os
import torch
from src import data_setup, engine, custom_resnet, utils
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt



# Setup hyperparameters
NUM_EPOCHS = 24
BATCH_SIZE = 512
LEARNING_RATE = 0.03
MOMENTUM = 0.9
MAX_LR = 5.22E-02
WEIGHT_DECAY = 1e-4

# Setup directories
train_dir = "../data"
test_dir = "../data"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
# Train Phase transformations
train_transforms = A.Compose([
    #A.HorizontalFlip(),
    #A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5),
    #A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=(0.49139968, 0.48215827, 0.44653124), mask_fill_value=None),  # Apply coarse dropout
    A.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),  # Normalize the image
    ToTensorV2() # Convert image to a PyTorch tensor
])


# Test Phase transformations
test_transforms = A.Compose([
    A.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768]),  # Normalize the image
    ToTensorV2()  # Convert image to a PyTorch tensor
])


# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    batch_size=BATCH_SIZE
)

# Create model with help from custom_resnet.py
model = custom_resnet.CustomResNet().to(device)

# Set loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_dataloader, end_lr=10, num_iter=200, step_mode="exp")
lr_finder.plot() # to inspect the loss-learning rate graph
plt.savefig("results/lr_finder_plot.png")
lr_finder.reset() # to reset the model and optimizer to their initial state

scheduler = OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=len(train_dataloader),
        epochs=NUM_EPOCHS,
        pct_start=5/NUM_EPOCHS,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
    )

# Start training with help from engine.py
engine.train(model=model,
             train_loader=train_dataloader,
             test_loader=test_dataloader,
             criterion=criterion,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device,
             scheduler=scheduler)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="results",
                 model_name="CustomResNet.pth")
