import utils    
import argparse
import os

data_path = "Data/TeleoSplitGenera_300_medium/"


# https://github.com/jerryji1993/DNABERT?tab=readme-ov-file#32-download-pre-trained-dnabert
# tokenizer = "./Model/3-new-12w-0/"
# model= "./Model/3-new-12w-0/"


# tokenizer = "zhihan1996/DNABERT-2-117M"
# model = "zhihan1996/DNABERT-2-117M"

tokenizer = 4
model = "zhihan1996/DNABERT-2-117M"

output_path = f"./Model/TeleoSplitGenera_300_medium_4mer_DNABERT-2-117M"

output_dir=output_path
learning_rate=1e-5
per_device_train_batch_size=16
per_device_eval_batch_size=16
num_train_epochs=3
weight_decay=0.01
eval_strategy="epoch"
save_strategy="epoch"
load_best_model_at_end=True
metric_for_best_model="f1"
greater_is_better=True
push_to_hub=False







def train(path, tokenizer, model):



    train, test, val = utils.load_dataset(path)
    tokenizer = utils.load_tokenizer(tokenizer)
    train_dataset, val_dataset, test_dataset, id2label, label2id= utils.encode_data(tokenizer, train, val, test)
    model = utils.load_model(model,id2label=id2label, label2id=label2id).to("cuda")
    arg_train= utils.training_argument(
        output_path=output_path,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        push_to_hub=push_to_hub
    )
    trainer = utils.define_trainer(model, tokenizer, train_dataset, val_dataset, arg_train)
    trainer.train()

    checkpoints = [os.path.join(output_dir, f) for f in os.listdir(output_dir)] # add path to each file
    checkpoints.sort(key=lambda x: os.path.getmtime(x))
    utils.plot_save_loss(checkpoints[-1])
    

if __name__=="__main__":

    train(data_path, tokenizer, model)


