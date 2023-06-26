import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir, test_dir, train_transforms, test_transforms, batch_size, num_workers=NUM_WORKERS):

  # Download the dataset
  train_data = datasets.CIFAR10(train_dir, train=True, download=True, transform=train_transforms)
  test_data = datasets.CIFAR10(test_dir, train=False, download=True, transform=test_transforms) 

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
