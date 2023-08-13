"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.optim.lr_scheduler import OneCycleLR

class YOLOv3Lightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = YOLOv3(num_classes=config.NUM_CLASSES)
        self.loss_fn = YoloLoss()
        self.scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(config.DEVICE)
        self.train_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y0, y1, y2 = (
            y[0],
            y[1],
            y[2],
        )
        out = self(x)
        loss = (
                self.loss_fn(out[0], y0, self.scaled_anchors[0])
                + self.loss_fn(out[1], y1, self.scaled_anchors[1])
                + self.loss_fn(out[2], y2, self.scaled_anchors[2])
        )
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)  # Logging the training loss for visualization
        self.train_step_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        print(f"\nCurrently epoch {self.current_epoch}")
        train_epoch_average = torch.stack(self.train_step_outputs).mean()
        self.train_step_outputs.clear()
        print(f"Train loss {train_epoch_average}")
        print("On Train Eval loader:")
        print("On Train loader:")
        class_accuracy, no_obj_accuracy, obj_accuracy = check_class_accuracy(self.model, self.train_loader, threshold=config.CONF_THRESHOLD)
        self.log("class_accuracy", class_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log("no_obj_accuracy", no_obj_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log("obj_accuracy", obj_accuracy, on_epoch=True, prog_bar=True, logger=True)

        if (self.current_epoch>0) and ((self.current_epoch+1) % 10 == 0):
            plot_couple_examples(self.model, self.test_loader, 0.6, 0.5, self.scaled_anchors)

        if (self.current_epoch>0) and (self.current_epoch+1 == 40):
            check_class_accuracy(self.model, self.test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                self.test_loader,
                self.model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")

            self.log("MAP", mapval.item(), on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        self.trainer.fit_loop.setup_data()
        dataloader = self.trainer.train_dataloader

        EPOCHS = config.NUM_EPOCHS
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=1.10E-01,
            steps_per_epoch=len(dataloader),
            epochs=EPOCHS,
            pct_start=5/EPOCHS,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy='linear'
        )

        scheduler = {"scheduler": lr_scheduler, "interval" : "step"}

        return [optimizer], [scheduler]

    def setup(self, stage=None):
        self.train_loader, self.test_loader, self.train_eval_loader = get_loaders(
            train_csv_path=self.config.DATASET + "/train.csv",
            test_csv_path=self.config.DATASET + "/test.csv",
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.train_eval_loader

    def test_dataloader(self):
        return self.test_loader


if __name__ == "__main__":

    model = YOLOv3Lightning(config)

    checkpoint = ModelCheckpoint(filename='last_epoch', save_last=True)
    lr_rate_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
                  max_epochs=config.NUM_EPOCHS,
                  deterministic=False,
                  logger=True,
                  callbacks=[checkpoint, lr_rate_monitor],
                  enable_model_summary=False,
                  log_every_n_steps=1,
                  precision=16
              )
    print("Training Started by Selvaraj Sembulingam")
    trainer.fit(model)
    torch.save(model.state_dict(), 'YOLOv3.pth')
