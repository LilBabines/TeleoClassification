import hydra
from omegaconf import DictConfig
import os
import sys 
import pandas as pd
sys.path.append(os.path.join(os.getcwd(), 'src'))
from models.tokenizer import load_tokenizer
from data.dataset import load_data, encode_multiTaxa_dataset, encode_singleTaxa_dataset
from models.model import MultiTaxaClassification, load_bert_model
from utils.trainer import define_trainer
from transformers import TrainingArguments, AutoModel
import torch
from safetensors import safe_open
from utils.visualize import plot_save_loss


# TODO: Add diferent loss, BCEWithLogitsLoss for weight imbalance classes
# TODO: -----------------, HierarchicalLoss for pénaliser les famille  qui ne sont pas dans l'ordre et que l'ordre est bien prédit
# TODO: Add bertax model............. je vais encore m'amuser moi
@hydra.main(version_base="1.3",config_path="config", config_name="config")
def main(cfg: DictConfig):
    
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print("log_dir: ", log_dir)
    
    tokenizer = load_tokenizer(cfg.model.tokenizer_name)

    if cfg.task.task == "multiTaxa":
        
        train_dataset, val_dataset, test_dataset, id2label_order, label2id_order, id2label_family, label2id_family = encode_multiTaxa_dataset(tokenizer, cfg.data.dataset_path)
        num_classes = (len(id2label_order) , len(id2label_family))
        model = MultiTaxaClassification( len(id2label_order), len(id2label_family),vocab_size = tokenizer.vocab_size,**cfg.model.bert_kwargs)  
    elif cfg.task.task == "singleTaxa":

        train_dataset, val_dataset, test_dataset, id2label, label2id = encode_singleTaxa_dataset(tokenizer,cfg.data.dataset_path )
        num_classes = len(id2label)
        model = load_bert_model(cfg.model.model_name, tokenizer.vocab_size, local=cfg.model.local, id2label=id2label, label2id=label2id)
    else:
        raise ValueError("cfg.task.task has to be either 'multiTaxa' or 'singleTaxa'")
    
    if not cfg.task.train :
        
        with safe_open(cfg.task.checkpoint_path+'/model.safetensors', framework="pt", device="cpu") as f:
            state_dict = {key: f.get_tensor(key) for key in f.keys()}
        model.load_state_dict(state_dict)   

    args = TrainingArguments(output_dir=log_dir,**cfg.trainer)
    trainer, metrics_order, metrics_family = define_trainer(model, tokenizer, train_dataset, val_dataset, num_classes,cfg.metrics,args)

    if cfg.task.train :

        trainer.train()
        plot_save_loss(log_dir, metrics = metrics_order +metrics_family)
    
        
    result = trainer.predict(test_dataset)
    
    if cfg.task.save_preds :

        if cfg.task.task == "multiTaxa":
            dataframe = pd.DataFrame( columns = ["preds_order","preds_family", "labels_order", "labels_family"]) 
            dataframe["preds_order"] = result.predictions[0].argmax(axis=1).squeeze()
            dataframe["preds_family"] = result.predictions[1].argmax(axis=1).squeeze()
            dataframe["labels_order"] = result.label_ids[:,0]
            dataframe["labels_family"] = result.label_ids[:,1]
            # dataframe["labels"] = result.label_ids
            # torch.save(result.predictions, os.path.join(log_dir,"predictions.pt"))
            # torch.save(result.label_ids, os.path.join(log_dir,"labels.pt"))
            dataframe.to_csv(os.path.join(log_dir,"predictions.csv"), index=False)
        else : 
            print("Saving predictions for singleTaxa not implemented yet")
    
    

if __name__ == "__main__":
    
    main()
    
    # d= ['macro_accuracy_order', 'micro_accuracy_order', 'macro_accuracy_family', 'micro_accuracy_family']
    # plot_save_loss(r"C:\Users\Auguste Verdier\Desktop\TeleoClassification\outputs\TeleoSplitGenera_300_medium\DNABERT-2-117\multiTaxa", metrics = d)
    