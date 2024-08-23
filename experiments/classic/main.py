import hydra
from omegaconf import DictConfig
import os
import sys 
sys.path.append(os.path.join(os.getcwd(), 'src'))
from models.tokenizer import load_tokenizer
from data.dataset import load_data, encode_multiTaxa_dataset, encode_singleTaxa_dataset
from models.model import MultiTaxaClassification, load_bert_model
from utils.trainer import define_trainer
from transformers import TrainingArguments
from utils.visualize import plot_save_loss

@hydra.main(version_base="1.3",config_path="config", config_name="config")
def main(cfg: DictConfig):
    
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print("log_dir: ", log_dir)
    
    tokenizer = load_tokenizer(cfg.model.tokenizer_name)

    if cfg.task.task == "multiTaxa":
        
        train_dataset, val_dataset, test_dataset, id2label_order, label2id_order, id2label_family, label2id_family = encode_multiTaxa_dataset(tokenizer, cfg.data.dataset_path)
        num_classes = (len(id2label_order) , len(id2label_family))
        model = MultiTaxaClassification( len(id2label_order), len(id2label_family))  
    elif cfg.task.task == "singleTaxa":
        train, test, val = load_data(cfg.data.dataset_path)
        train_dataset, val_dataset, test_dataset, id2label, label2id = encode_singleTaxa_dataset(tokenizer,cfg.data.dataset_path )
        num_classes = len(id2label)
        model = load_bert_model(cfg.model.model_name, tokenizer.vocab_size, local=cfg.model.local, id2label=id2label, label2id=label2id)
    else:
        raise ValueError("cfg.task.task has to be either 'multiTaxa' or 'singleTaxa'")
    
    args = TrainingArguments(output_dir=log_dir,**cfg.trainer)
    trainer = define_trainer(model, tokenizer, train_dataset, val_dataset, num_classes,cfg.metrics,args)
    
    trainer.train()
    # plot_save_loss(log_dir+os.listdir(log_dir)[-1])

if __name__ == "__main__":
    main()
    # plot_save_loss(r'outputs\main\2024-08-23_15-52-05\checkpoint-19640')