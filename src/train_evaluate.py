import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score
import torch


def train_model(model, train_data_loader, val_data_loader, epochs, lr, wandb, device="cpu"):
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print('-' * 10)

        train_loss = 0.0
        val_loss = 0.0

        model = model.train()

        for step, batch in enumerate(train_data_loader):
            all_preds = []
            all_labels = []
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            labels = batch["label"].to(device)

            loss, outputs = model(input_ids, attention_mask, labels)
            
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            acc_score = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)

            wandb.log({"step_train_accuracy": acc_score, "step_train_f1": f1})
            
            train_loss += loss.item()
            wandb.log({"step_train_loss": loss.item()})
            loss.backward()
            optimizer.step()
            
        acc_score, f1 = evaluate_model(model, train_data_loader, device)
        wandb.log({"epoch_train_accuracy": acc_score, "epoch_train_f1": f1})

        model = model.eval()

        with torch.no_grad():
            for step, batch in enumerate(val_data_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                loss, outputs = model(input_ids, attention_mask, labels)
                val_loss += loss.item()

        train_loss /= len(train_data_loader)
        val_loss /= len(val_data_loader)

        acc, f1 = evaluate_model(model, val_data_loader, device)

        wandb.log({"epoch_val_accuracy": acc, "epoch_val_f1": f1})
        
        wandb.log({"epoch_train_loss": train_loss, "epoch_val_loss": val_loss})
        print(f"Train loss {train_loss} validation loss {val_loss}")



def evaluate_model(model, data_loader, device="cpu"):
    model = model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc_score = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return acc_score, f1
