import torch
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import AutoModelForSequenceClassification
from torchmetrics import Accuracy, F1Score as F1

class ArgumentModel(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # freeze all parameters except the last layer
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        self.loss_fn = CrossEntropyLoss()
        self.lr = lr

        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.train_f1 = F1(task='binary')
        self.val_f1 = F1(task='binary')
        
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        if labels is not None:
            loss = self.loss_fn(output.logits, labels)
            return loss, output
        else:
            return output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        loss, outputs = self(input_ids, attention_mask, labels)
        preds = torch.softmax(outputs.logits)
        self.train_accuracy.update(preds, labels)
        self.train_f1.update(preds, labels)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        loss, outputs = self(input_ids, attention_mask, labels)
        preds = torch.softmax(outputs.logits)
        self.val_accuracy.update(preds, labels)
        self.val_f1.update(preds, labels)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outs):
        # Log metrics at the end of each epoch
        self.log('train_acc', self.train_accuracy.compute(), on_epoch=True, prog_bar=True)
        self.log('train_f1', self.train_f1.compute(), on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outs):
        # Log metrics at the end of each epoch
        self.log('val_acc', self.val_accuracy.compute(), on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1.compute(), on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
