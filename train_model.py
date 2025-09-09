import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from ner_dataset import NERDataset
from dvclive.lightning import DVCLiveLogger
import dvc.api
from model import NERModel

params = dvc.api.params_show()


class NERDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=16):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=params["train"]["num-workers"]
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=params["train"]["num-workers"]
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=params["train"]["num-workers"]
        )

    def collate_fn(self, batch):
        # Паддинг батча
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
        labels = [torch.tensor(item['labels']) for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

if __name__ == "__main__":

    train_dataset:NERDataset = torch.load('data/ner_dataset/train_dataset.pt', weights_only=False)
    val_dataset:NERDataset = torch.load('data/ner_dataset/val_dataset.pt', weights_only=False)
    test_dataset:NERDataset = torch.load('data/ner_dataset/test_dataset.pt', weights_only=False)

    with open('data/ner_dataset/labels.txt', 'r') as f:
        label_list = [line.strip() for line in f]

    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    data_module = NERDataModule(train_dataset, val_dataset, test_dataset, batch_size=params["train"]["batch-size"])

    model = NERModel(
        model_name=params["base-model"],
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        learning_rate=params["train"]["lr"],
        freeze=params["train"]["freeze"],
        dropout_rate= params["train"]["dropout_rate"],
        use_prev_label= params["train"]["use_prev_label"],
    )

    trainer = pl.Trainer(
        max_epochs=params["train"]["epoches"],
        accelerator="gpu",
        logger=DVCLiveLogger(log_model=True),
        num_sanity_val_steps=10,
    )

    trainer.fit(model, data_module)

    trainer.test(model, data_module)

    import os
    os.makedirs("models", exist_ok=True)
    
    torch.save(model.state_dict(), f"models/model_{params["base-model"].split("/")[-1]}_{params["train"]["epoches"]}_eps.pth")
