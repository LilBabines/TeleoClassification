import utils    
import argparse
import os

data_path = "Data/TeleoSplitGenera_300_medium/"
tokenizer = "./Model/3-new-12w-0/"
mode= "./Model/3-new-12w-0/"
output_path = f"./Model/train_{data_path.split('/')[-2]}_{tokenizer.split('/')[-2]}_{mode.split('/')[-2]}"

def train(path, tokenizer, name):
    train, test, val = utils.load_dataset(path)
    tokenizer = utils.load_tokenizer(tokenizer)
    train_dataset, val_dataset, test_dataset, id2label, label2id= utils.encode_data(tokenizer, train, val, test)
    model = utils.load_model(name,id2label=id2label, label2id=label2id)
    trainer = utils.define_trainer(model, tokenizer, train_dataset, val_dataset,output_path=output_path)
    # trainer.train()
    

if __name__=="__main__":

    train(data_path, tokenizer, mode)


