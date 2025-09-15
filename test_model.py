from train_model import params, NERDataModule, NERDataset, torch, NERModel, np, pl, DVCLiveLogger

if __name__ == "__main__":
        
        with open('data/ner_dataset/labels.txt', 'r') as f:
            label_list = [line.strip() for line in f]
        
        id2label = {i: label for i, label in enumerate(label_list)}
        label2id = {label: i for i, label in enumerate(label_list)}

        weigts = np.array(params["train"]["weights"])
        weigts /= weigts.sum()

        model = NERModel(
        model_name=params["base-model"],
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        learning_rate=params["train"]["lr"],
        freeze=params["train"]["freeze"],
        dropout_rate= params["train"]["dropout_rate"],
        use_prev_label= params["train"]["use_prev_label"],
        weights= weigts.tolist()
        )
        model.load_state_dict(torch.load("models/model.pth"))
        model.eval()
        test_dataset:NERDataset = torch.load('data/ner_dataset/test_dataset.pt', weights_only=False)

        trainer = pl.Trainer(
        max_epochs=params["train"]["epoches"],
        accelerator="gpu",
        logger=DVCLiveLogger(log_model=True),
        num_sanity_val_steps=10,
        )
        data_module = NERDataModule(None, None, test_dataset, batch_size=params["train"]["batch-size"])

        trainer.test(model, data_module)
