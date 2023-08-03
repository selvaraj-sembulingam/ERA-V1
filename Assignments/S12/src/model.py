import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl

class CustomResNet(pl.LightningModule):
    def __init__(self, dropout_value=0.01):
        super(CustomResNet, self).__init__()

        self.test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': [], 'grad_cam': []}

        self.preplayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.resblock1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, bias=False, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )
        self.resblock2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, bias=False, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
            )
        self.maxpoollayer = nn.Sequential(
            nn.MaxPool2d(kernel_size = 4, stride = 4)
            )
        self.fclayer = nn.Sequential(
            nn.Linear(512,10)
            )
        
        self.accuracy = Accuracy(task='multiclass', num_classes=10)
        
    def loss_function(self, pred, target):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(pred, target)
        
    def forward(self, x):
        x = self.preplayer(x)
        x = self.layer1(x)
        r1 = self.resblock1(x)
        x = x + r1
        x = self.layer2(x)
        x = self.layer3(x)
        r2 = self.resblock2(x)
        x = x + r2
        x = self.maxpoollayer(x)
        x = x.view((x.shape[0],-1))
        x = self.fclayer(x)
        
        return x

    def get_misclassified_images(self, batch):
        images, labels = batch
        predictions = self(images)
        
        # Check for misclassifications and store the misclassified images
        predicted_labels = torch.argmax(predictions, dim=1)
        misclassified_mask = predicted_labels != labels
        misclassified_images = images[misclassified_mask]

        # Append the misclassified images to the list
        self.test_incorrect_pred['images'].extend(misclassified_images)
        self.test_incorrect_pred['ground_truths'].extend(labels)
        self.test_incorrect_pred['predicted_vals'].extend(predicted_labels)



    def save_misclassified_images(self, images, labels, predictions, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for i in range(20):
            img = images[i].permute(1, 2, 0).cpu().numpy()
            true_label = labels[i].item()
            pred_label = predictions[i].item()

            filename = f"misclassified_{i}_true_{true_label}_pred_{pred_label}.png"
            filepath = os.path.join(output_dir, filename)

            plt.imshow(img)
            plt.title(f"True Label: {true_label}, Predicted Label: {pred_label}")
            plt.axis("off")
            plt.savefig(filepath, bbox_inches="tight")
            plt.close()


    def get_loss_accuracy(self, batch):
        images, labels = batch
        predictions = self(images)
        loss = self.loss_function(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.get_loss_accuracy(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.get_loss_accuracy(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

        return loss


    def test_step(self, batch, batch_idx):
        loss = self.validation_step(batch, batch_idx)
        self.get_misclassified_images(batch)
        return loss

    def on_test_end(self):
        # After testing is complete, save the misclassified images
        images = torch.stack(self.test_incorrect_pred['images'])
        labels = torch.stack(self.test_incorrect_pred['ground_truths'])
        predictions = torch.stack(self.test_incorrect_pred['predicted_vals'])

        output_dir = "misclassified_images"
        self.save_misclassified_images(images, labels, predictions, output_dir)

        # Clear the list for the next testing
        self.test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}



    def configure_optimizers(self):
        LEARNING_RATE=0.03
        WEIGHT_DECAY=0
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        self.trainer.fit_loop.setup_data()
        dataloader = self.trainer.train_dataloader

        lr_scheduler = OneCycleLR(
          optimizer,
          max_lr=4.79E-02,
          steps_per_epoch=len(dataloader),
          epochs=24,
          pct_start=5/24,
          div_factor=100,
          three_phase=False,
          final_div_factor=100,
          anneal_strategy='linear'
        )

        scheduler = {"scheduler": lr_scheduler, "interval" : "step"}

        return [optimizer], [scheduler]
