import torch
import pytorch_lightning as pl
from transformers import AutoModelForTokenClassification
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


class NERModel(pl.LightningModule):
    def __init__(self, model_name, num_labels, id2label, label2id, learning_rate=2e-5, freeze=True, 
                 hidden_size=256, dropout_rate=0.3, use_prev_label=True):
        super().__init__()
        self.save_hyperparameters()
        
        self.use_prev_label = use_prev_label
        self.num_labels = num_labels
        
        self.bert = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        ).bert
        self.bert.train()
        
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
            
            for layer in self.bert.encoder.layer[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        bert_hidden_size = self.bert.config.hidden_size
        
        if self.use_prev_label:
            self.label_embedding = nn.Linear(num_labels+1, 32)  
            classifier_input_size = bert_hidden_size + 32
        else:
            classifier_input_size = bert_hidden_size
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.learning_rate = learning_rate
        
        self._init_metrics(num_labels)
        
        self.start_label_id = num_labels  # Специальный ID для начала последовательности

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def _init_metrics(self, num_labels):
        # Макро-метрики
        self.train_precision = MulticlassPrecision(num_classes=num_labels, average='macro', ignore_index=-100)
        self.train_recall = MulticlassRecall(num_classes=num_labels, average='macro', ignore_index=-100)
        self.train_f1 = MulticlassF1Score(num_classes=num_labels, average='macro', ignore_index=-100)

        self.val_precision = MulticlassPrecision(num_classes=num_labels, average='macro', ignore_index=-100)
        self.val_recall = MulticlassRecall(num_classes=num_labels, average='macro', ignore_index=-100)
        self.val_f1 = MulticlassF1Score(num_classes=num_labels, average='macro', ignore_index=-100)

        self.test_precision = MulticlassPrecision(num_classes=num_labels, average='macro', ignore_index=-100)
        self.test_recall = MulticlassRecall(num_classes=num_labels, average='macro', ignore_index=-100)
        self.test_f1 = MulticlassF1Score(num_classes=num_labels, average='macro', ignore_index=-100)

        # Per-label метрики
        self.train_precision_per_label = MulticlassPrecision(num_classes=num_labels, average=None, ignore_index=-100)
        self.train_recall_per_label = MulticlassRecall(num_classes=num_labels, average=None, ignore_index=-100)
        self.train_f1_per_label = MulticlassF1Score(num_classes=num_labels, average=None, ignore_index=-100)

        self.val_precision_per_label = MulticlassPrecision(num_classes=num_labels, average=None, ignore_index=-100)
        self.val_recall_per_label = MulticlassRecall(num_classes=num_labels, average=None, ignore_index=-100)
        self.val_f1_per_label = MulticlassF1Score(num_classes=num_labels, average=None, ignore_index=-100)

        self.test_precision_per_label = MulticlassPrecision(num_classes=num_labels, average=None, ignore_index=-100)
        self.test_recall_per_label = MulticlassRecall(num_classes=num_labels, average=None, ignore_index=-100)
        self.test_f1_per_label = MulticlassF1Score(num_classes=num_labels, average=None, ignore_index=-100)


    def forward(self, input_ids, attention_mask, labels=None, prev_labels_onehot=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        if self.use_prev_label and prev_labels_onehot is None:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            
            preds = torch.full((batch_size, seq_len), self.start_label_id, 
                              dtype=torch.long, device=device)
            logits = torch.full((batch_size, seq_len, self.num_labels), 0.0, device=device)

            for i in range(seq_len):
                current_prev_labels_onehot = F.one_hot(preds, num_classes=self.num_labels+1).float()
                
                label_embeddings = self.label_embedding(current_prev_labels_onehot)
                
                combined_output = torch.cat([sequence_output, label_embeddings], dim=-1)
                
                l = self.classifier(combined_output)
                logits[:, i] = l[:, i]
                current_preds = torch.argmax(l[:, i, :], dim=-1)
                preds[:, i] = current_preds
        
        elif self.use_prev_label and prev_labels_onehot is not None:
            label_embeddings = self.label_embedding(prev_labels_onehot)
            
            combined_output = torch.cat([sequence_output, label_embeddings], dim=-1)
            logits = self.classifier(combined_output)
        else:
            logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:

            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {"loss": loss, "logits": logits}

    def _log_metrics_per_label(self, phase, precision_metric, recall_metric, f1_metric):
        precision_scores = precision_metric.compute()
        recall_scores = recall_metric.compute()
        f1_scores = f1_metric.compute()
        
        for idx, (prec, rec, f1) in enumerate(zip(precision_scores, recall_scores, f1_scores)):
            label_name = self.hparams.id2label[idx]
            self.log(f'{phase}_precision_{label_name}', prec)
            self.log(f'{phase}_recall_{label_name}', rec)
            self.log(f'{phase}_f1_{label_name}', f1)

    def _get_prev_labels_onehot(self, labels):
        batch_size, seq_len = labels.shape
        prev_labels = torch.cat([
            torch.full((batch_size, 1), self.start_label_id, dtype=torch.long, device=labels.device),
            labels[:, :-1]
        ], dim=1)

        prev_labels[prev_labels == -100] = self.start_label_id
        prev_labels_onehot = F.one_hot(prev_labels, num_classes=self.num_labels + 1).float()
        
        return prev_labels_onehot

    def training_step(self, batch, batch_idx):
        prev_labels_onehot = self._get_prev_labels_onehot(batch['labels'])
        
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            prev_labels_onehot=prev_labels_onehot
        )
        loss = outputs["loss"]

        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)


        self.train_precision(preds, batch['labels'])
        self.train_recall(preds, batch['labels'])
        self.train_f1(preds, batch['labels'])
        self.train_precision_per_label(preds, batch['labels'])
        self.train_recall_per_label(preds, batch['labels'])
        self.train_f1_per_label(preds, batch['labels'])

        self.log('train_loss', loss)
        self.log('train_precision_macro', self.train_precision, on_step=False, on_epoch=True)
        self.log('train_recall_macro', self.train_recall, on_step=False, on_epoch=True)
        self.log('train_f1_macro', self.train_f1, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        self._log_metrics_per_label('train', 
                                  self.train_precision_per_label,
                                  self.train_recall_per_label,
                                  self.train_f1_per_label)
        self.train_precision_per_label.reset()
        self.train_recall_per_label.reset()
        self.train_f1_per_label.reset()

    def validation_step(self, batch, batch_idx):

        prev_labels_onehot = self._get_prev_labels_onehot(batch['labels'])

        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            prev_labels_onehot=prev_labels_onehot
        )
        loss = outputs["loss"]

        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)


        self.val_precision(preds, batch['labels'])
        self.val_recall(preds, batch['labels'])
        self.val_f1(preds, batch['labels'])
        self.val_precision_per_label(preds, batch['labels'])
        self.val_recall_per_label(preds, batch['labels'])
        self.val_f1_per_label(preds, batch['labels'])

        self.log('val_loss', loss)
        self.log('val_precision_macro', self.val_precision, on_step=False, on_epoch=True)
        self.log('val_recall_macro', self.val_recall, on_step=False, on_epoch=True)
        self.log('val_f1_macro', self.val_f1, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        self._log_metrics_per_label('val',
                                  self.val_precision_per_label,
                                  self.val_recall_per_label,
                                  self.val_f1_per_label)
        
        self.val_precision_per_label.reset()
        self.val_recall_per_label.reset()
        self.val_f1_per_label.reset()

    def test_step(self, batch, batch_idx):
        
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            prev_labels_onehot=None
        )
        loss = outputs["loss"]

        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=-1)


        self.test_precision(preds, batch['labels'])
        self.test_recall(preds, batch['labels'])
        self.test_f1(preds, batch['labels'])
        self.test_precision_per_label(preds, batch['labels'])
        self.test_recall_per_label(preds, batch['labels'])
        self.test_f1_per_label(preds, batch['labels'])

        self.log('test_loss', loss)
        self.log('test_precision_macro', self.test_precision, on_step=False, on_epoch=True)
        self.log('test_recall_macro', self.test_recall, on_step=False, on_epoch=True)
        self.log('test_f1_macro', self.test_f1, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        self._log_metrics_per_label('test',
                                  self.test_precision_per_label,
                                  self.test_recall_per_label,
                                  self.test_f1_per_label)

        self.test_precision_per_label.reset()
        self.test_recall_per_label.reset()
        self.test_f1_per_label.reset()

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch["attention_mask"],
            labels=None,
            prev_labels_onehot=None
        )
        logits = outputs["logits"]
            
        preds = torch.argmax(logits, dim=-1)
        
        return preds
    