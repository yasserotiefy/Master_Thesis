import torch
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import AutoModelForSequenceClassification
from torchmetrics.classification import Accuracy, F1Score as F1


class ArgumentModel(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, force_download=True, num_labels=2, ignore_mismatched_sizes=True)
        # freeze all parameters except the last layer
        for param in self.model.base_model.parameters():
            param.requires_grad = False
        

        self.loss_fn = CrossEntropyLoss()
        self.lr = lr

        self.accuracy = Accuracy(task="multiclass", num_classes=2)
        self.f1 = F1(task="multiclass", num_classes=2, average="macro")
        # self.val_accuracy = Accuracy(task="multiclass", num_classes=2)
        # self.val_f1 = F1(task='multiclass', num_classes=2)

    def forward(self, input_ids, attention_mask, labels=None):
        labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits


        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        else:
            return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]


        labels = batch["label"]
        loss, logits = self.forward(input_ids, attention_mask, labels)
        preds = torch.argmax(logits, dim=1) # convert logits to 1-hot encoding
        
        # log this values to wandb
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc",
            self.accuracy(preds, labels),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_f1",
            self.f1(preds, labels),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        loss, logits = self.forward(input_ids, attention_mask, labels)
        pred = torch.argmax(logits, dim=1) # convert logits to 1-hot encoding

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy(pred, labels), prog_bar=True)
        self.log("val_f1", self.f1(pred, labels), prog_bar=True)

        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
