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
from utils.visualize import plot_save_loss

sys.path.append(os.path.join(os.getcwd(), 'resources'))
from dnabert2  import bert_layers

from transformers import TrainingArguments, AutoModel, EarlyStoppingCallback
import torch



# TODO: Add diferent loss, BCEWithLogitsLoss for weight imbalance classes
# TODO: -----------------, HierarchicalLoss for pénaliser les famille  qui ne sont pas dans l'ordre et que l'ordre est bien prédit
# TODO: Add bertax model............. je vais encore m'amuser moi

@hydra.main(version_base="1.3",config_path="config", config_name="config")
def main(cfg: DictConfig):
    
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print("log_dir: ", log_dir)
    
    tokenizer = load_tokenizer(cfg.model.tokenizer_name)
    print("len tokenizer: ", tokenizer.vocab_size)
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
    
    
    masked_lm_model = bert_layers.BertForMaskedLM.from_pretrained(cfg.model.local_path, trust_remote_code=True)
    
    model.bert.load_state_dict(masked_lm_model.bert.state_dict(),strict=False)

    args = TrainingArguments(output_dir=log_dir,**cfg.trainer.kwargs,report_to="tensorboard")
    trainer, metrics_order, metrics_family = define_trainer(model, tokenizer, train_dataset, val_dataset, num_classes,cfg.metrics,args) #,callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.trainer.early_stopping_patience)]

    if cfg.task.train :

        trainer.train()
        plot_save_loss(log_dir, metrics = metrics_order +metrics_family)
    
        
    result = trainer.predict(test_dataset)
    print("Metrics on test set: ", result.metrics)
    if cfg.task.save_preds :

        if cfg.task.task == "multiTaxa":
            dataframe = pd.DataFrame( columns = ["preds_order","preds_family", "labels_order", "labels_family"]) 
            dataframe["preds_order"] = result.predictions[0].argmax(axis=1).squeeze()
            dataframe["preds_family"] = result.predictions[1].argmax(axis=1).squeeze()
            
            dataframe["labels_order"] = result.label_ids[:,0]
            dataframe["labels_family"] = result.label_ids[:,1]


            import pickle

            
            pickle.dump({'preds':result.predictions,'labels':result.label_ids}, open(os.path.join(log_dir,"predictions.pkl"), 'wb'))
            dataframe.to_csv(os.path.join(log_dir,"predictions.csv"), index=False)
        else : 
            
            dataframe = pd.DataFrame( columns = ["preds_family", "labels_family"]) 
           
            dataframe["preds_family"] = result.predictions[0].argmax(axis=1).squeeze()
            
            
            dataframe["labels_family"] = result.label_ids


            import pickle

            
            pickle.dump({'preds':result.predictions,'labels':result.label_ids}, open(os.path.join(log_dir,"predictions.pkl"), 'wb'))
            dataframe.to_csv(os.path.join(log_dir,"predictions.csv"), index=False)
    # model.save_pretrained(log_dir)

if __name__ == "__main__":
    
    main()
    # pass
    # d= ['macro_accuracy_order', 'micro_accuracy_order', 'macro_accuracy_family', 'micro_accuracy_family']
    # plot_save_loss(r"C:\Users\Auguste Verdier\Desktop\TeleoClassification\outputs\TeleoSplitGenera_300_medium\DNABERT-2-117\multiTaxa", metrics = d)
    