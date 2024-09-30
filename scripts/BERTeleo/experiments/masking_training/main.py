import hydra
from omegaconf import DictConfig
import os

import sys 
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data.dataset import encode_self_supervised_dataset
from models.tokenizer import load_tokenizer
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

sys.path.append(os.path.join(os.getcwd(), 'resources'))
from data.dataset import encode_self_supervised_dataset
from dnabert2 import bert_layers


# TODO: Add diferent loss, BCEWithLogitsLoss for weight imbalance classes
# TODO: -----------------, HierarchicalLoss for pénaliser les famille  qui ne sont pas dans l'ordre et que l'ordre est bien prédit
# TODO: Add bertax model............. je vais encore m'amuser moi

@hydra.main(version_base="1.3",config_path="config", config_name="config")
def main(cfg: DictConfig):
    
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if cfg.data.kfold:

        tokenizer = load_tokenizer(cfg.model.tokenizer_name)

        for fold in os.listdir(cfg.data.dataset_path):


            train_dataset, val_dataset= encode_self_supervised_dataset(tokenizer,dir_path=os.path.join(cfg.data.dataset_path,fold))
            model = bert_layers.BertForMaskedLM.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
            model.resize_token_embeddings(len(tokenizer))
            
    
    
    
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm_probability= 0.20
                )

            args = TrainingArguments(output_dir=os.path.join(log_dir,'4mer','checkpoints',fold),**cfg.trainer.kwargs,save_safetensors=False, logging_dir = os.path.join(log_dir,'4mer','logs',fold))
            trainer = Trainer(
                model=model,
                args=args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                #callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
            )
            if cfg.task.train :
                trainer.evaluate()
                trainer.train()
        
        tokenizer = load_tokenizer("zhihan1996/DNABERT-2-117M")

        for fold in os.listdir(cfg.data.dataset_path):


            train_dataset, val_dataset= encode_self_supervised_dataset(tokenizer,dir_path=os.path.join(cfg.data.dataset_path,fold))
            model = bert_layers.BertForMaskedLM.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
            model.resize_token_embeddings(len(tokenizer))
            
    
    
    
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm_probability= 0.20
                )

            args = TrainingArguments(output_dir=os.path.join(log_dir,'dnabert2_tkz','checkpoints',fold),**cfg.trainer.kwargs,save_safetensors=False, logging_dir = os.path.join(log_dir,'dnabert2_tkz','logs',fold))
            trainer = Trainer(
                model=model,
                args=args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                #callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
            )
            if cfg.task.train :
                trainer.evaluate()
                trainer.train()

if __name__ == "__main__":
    
    main()
