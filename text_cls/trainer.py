import torch
import numpy as np

from torch import nn
from transformers import Trainer
from sklearn.metrics import mean_squared_error


class CETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.get("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class WCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.get("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Get tensor device
        device = labels.get_device()
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # For Set 1, multi-task: [0.96, 0.86, 0.66, 0.68, 0.85, 0.98]
        # For Set 2, multi-task: [0.84, 0.74, 0.62, 0.83, 0.98]
        if self.model.config.num_labels == 6:
            weight = torch.tensor([0.96, 0.86, 0.66, 0.68, 0.85, 0.98]).to(device)
        else:
            weight = torch.tensor([0.84, 0.74, 0.62, 0.83, 0.98]).to(device)
        print(f"weight.get_device(): {weight.get_device()}")
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class MSETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.get("labels")

        # forward pass
        # model = model.to(device)    # TODO: Check necessary or not
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.MSELoss()
        loss = loss_fct(logits.view(-1), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


class MixTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.get("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        gold_labels = labels.tolist()
        probs = torch.softmax(logits, dim=-1)
        pred_labels = [np.argmax(pred) for pred in probs.tolist()]

        print(f"labels: {labels}")
        print(f"pred_labels: {pred_labels}")

        loss_mse = torch.tensor(
            mean_squared_error(gold_labels, pred_labels), requires_grad=True
        )

        print(f"loss_mse: {loss_mse}")

        loss += loss_mse
        print(f"loss: {loss}")

        return (loss, outputs) if return_outputs else loss
