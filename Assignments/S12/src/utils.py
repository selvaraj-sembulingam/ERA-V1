import torch
from src.model import CustomResNet
import numpy as np
from src.datamodule import CIFARDataModule
import matplotlib.pyplot as plt
import seaborn as sn
import math
import cv2

device = "cpu"

def load_model():
  model = CustomResNet()
  model.load_state_dict(torch.load('CustomResNet.pth', map_location=torch.device('cpu')), strict=False)
  model = model.to(device)
  model.eval()
  model.freeze()
  return model

def get_misclassified_images():
    data_module = CIFARDataModule(batch_size=1)
    data_module.setup()

    error_images = []
    error_images_gradcam = []
    error_label = []
    error_pred = []
    error_prob = []

    device = "cpu"
    i = 1
    for batch in data_module.val_dataloader():
        x, y = batch
        model = load_model()
        x = x.to(device)
        y = y.to(device)
        _X_valid, _y_valid =x, y
        pred = torch.softmax(model(_X_valid), axis=-1).cpu().numpy()
        pred_class = pred.argmax(axis=-1)
        if pred_class != _y_valid.cpu().numpy():
            error_images.extend(_X_valid)
            error_label.extend(_y_valid)
            error_pred.extend(pred_class)
            error_prob.extend(pred.max(axis=-1)) 
            i+=1
            if i>20:
              break
    return error_images, error_label, error_pred, error_prob

def denormalize_image(image):
    return image * [0.24703233, 0.24348505, 0.26158768] + [0.49139968, 0.48215827, 0.44653124]

classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

def save_misclassified_images(error_images, error_label, error_pred, error_prob):
  plt.figure()
  for ind, image in enumerate(error_images):
      # plt.subplot(h_size, w_size, ind + 1)
      plt.imshow(denormalize_image(image.permute(1, 2, 0).cpu().numpy()))
      pred_label = classes[error_pred[ind]]
      pred_prob = error_prob[ind]
      true_label = classes[error_label[ind]]
      plt.title(f'predict: {pred_label} ({pred_prob:.2f}) true: {true_label}')
      plt.axis('off')
      plt.imsave("misclassified"+str(ind)+".jpg",denormalize_image(image.permute(1, 2, 0).cpu().numpy()))
      plt.clf()

def plot_misclassified_images(error_images, error_label, error_pred, error_prob, grad_cam=False):
    num_images = len(error_images)
    h_size, w_size = [4, 5]
    plt.figure(figsize=(15, 15))
    
    for ind, image in enumerate(error_images):
        plt.subplot(h_size, w_size, ind + 1)
        pred_label = classes[error_pred[ind]]
        pred_prob = error_prob[ind]
        true_label = classes[error_label[ind]]
        if not grad_cam:
          plt.imshow(denormalize_image(image.permute(1, 2, 0).cpu().numpy()))

        plt.title(f'GT: {true_label}, Pred: {pred_label}')
        plt.axis('off')
        plt.imsave(f"misclassified{ind}.jpg", denormalize_image(image.permute(1, 2, 0).cpu().numpy()))
    
    plt.tight_layout()
    plt.show()
