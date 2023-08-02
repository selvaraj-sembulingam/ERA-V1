if __name__ == '__main__':

  torch.manual_seed(1)
  pl.seed_everything(1, workers=True)

  data_module = CIFARDataModule()
  model = CustomResNet()

  checkpoint = ModelCheckpoint(monitor="acc/val", mode="max", save_last=True)
  lr_rate_monitor = LearningRateMonitor(logging_interval="epoch")

  trainer = pl.Trainer(
      max_epochs=24,
      deterministic=True,
      logger=True,
      callbacks=[checkpoint, lr_rate_monitor],
      enable_model_summary=False,
      log_every_n_steps=1,
      precision=16
  )
  trainer.fit(model, data_module)
