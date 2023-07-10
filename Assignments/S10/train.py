import os
import torch
from src import data_setup, engine, custom_resnet, utils
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms

torch.manual_seed(1)

# Setup hyperparameters
NUM_EPOCHS = 24
BATCH_SIZE = 512
LEARNING_RATE = 0.03
MOMENTUM = 0.9
MAX_LR = 4.79E-02
WEIGHT_DECAY = 1e-4

# Setup directories
train_dir = "../data"
test_dir = "../data"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
# Train Phase transformations
train_transforms = A.Compose([
    A.PadIfNeeded(min_height=32 + 4, min_width=32 + 4, p=1),
    A.RandomCrop(32, 32),
    A.HorizontalFlip(),
    A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=(0.49139968, 0.48215827, 0.44653124), mask_fill_value=None),  # Apply coarse dropout
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
