import hydra
from omegaconf import DictConfig
import os

# import sys 
# sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.data.dataset import encode_self_supervised_dataset
from src.models.tokenizer import load_tokenizer
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from data.dataset import encode_self_supervised_dataset
from resources.dnabert2 import bert_layers


# TODO: Add diferent loss, BCEWithLogitsLoss for weight imbalance classes
# TODO: -----------------, HierarchicalLoss for pénaliser les famille  qui ne sont pas dans l'ordre et que l'ordre est bien prédit
# TODO: Add bertax model............. je vais encore m'amuser moi

@hydra.main(version_base="1.3",config_path="config", config_name="config")
def main(cfg: DictConfig):
    
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print("log_dir: ", log_dir)
    
    tokenizer = load_tokenizer(cfg.model.tokenizer_name)

    if cfg.task.task == "masking":
        
        train_dataset, val_dataset= encode_self_supervised_dataset(tokenizer,dir_path='experiments/masking_training/data/')
        model = bert_layers.BertForMaskedLM.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    
    else : 
        print(" ITS A MASKING TASK, check the config file")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability= 0.20
        )

    args = TrainingArguments(output_dir=os.path.join(log_dir,'checkpoints'),**cfg.trainer.kwargs)
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

        trainer.train()
        
    

if __name__ == "__main__":
    
    main()
