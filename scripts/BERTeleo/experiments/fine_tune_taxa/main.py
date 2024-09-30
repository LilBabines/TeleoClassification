import hydra
from omegaconf import DictConfig
import os
import sys 
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), 'src'))
from models.tokenizer import load_tokenizer
from data.dataset import load_data, encode_multiTaxa_dataset, encode_singleTaxa_dataset
from models.model import MultiTaxaClassification, load_bert_model, get_best
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

    for fold in os.listdir(cfg.data.dataset_path):

        print('-----------------')
        print(f"Fold: {fold}")
        if os.path.exists(os.path.join(log_dir,'checkpoints',fold)):
            print("Fold already trained")
            continue
        print('-----------------')
        print("Model Initialization")
        if cfg.task.task == "multiTaxa":
            
            train_dataset, val_dataset, test_dataset, id2label_order, label2id_order, id2label_family, label2id_family = encode_multiTaxa_dataset(tokenizer, os.path.join(cfg.data.dataset_path,fold))
            num_classes = (len(id2label_order) , len(id2label_family))

            model = MultiTaxaClassification( len(id2label_order), len(id2label_family),vocab_size = tokenizer.vocab_size,**cfg.model.bert_kwargs)  
        elif cfg.task.task == "singleTaxa":

            train_dataset, val_dataset, test_dataset, id2label, label2id = encode_singleTaxa_dataset(tokenizer,os.path.join(cfg.data.dataset_path,fold))
            num_classes = len(id2label)
            model = load_bert_model(cfg.model.model_name, tokenizer.vocab_size, local=cfg.model.local, id2label=id2label, label2id=label2id)
        else:
            raise ValueError("cfg.task.task has to be either 'multiTaxa' or 'singleTaxa'")
        
        best_model = get_best(os.path.join(cfg.model.local_path,fold))
        print("best_model: ", best_model)
        
        masked_lm_model = bert_layers.BertForMaskedLM.from_pretrained(best_model)
        
        
        model.bert.load_state_dict(masked_lm_model.bert.state_dict(),strict=False)

        args = TrainingArguments(os.path.join(log_dir,'checkpoints',fold),
                                 **cfg.trainer.kwargs,
                                 save_safetensors=False, 
                                 logging_dir = os.path.join(log_dir,'logs',fold))
        trainer, metrics_order, metrics_family = define_trainer(model, tokenizer, train_dataset, val_dataset, num_classes,cfg.metrics,args,callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.trainer.early_stopping_patience)]) #

        print("Training")
        if cfg.task.train :

            trainer.train()
            plot_save_loss(os.path.join(log_dir,'checkpoints',fold), metrics = metrics_order +metrics_family)
        
            
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
                pickle.dump({'preds':result.predictions,'labels':result.label_ids}, open(os.path.join(log_dir,'checkpoints',fold,"predictions.pkl"), 'wb'))
                dataframe.to_csv(os.path.join(log_dir,'checkpoints',fold,"predictions.csv"), index=False)

            else : 
                
                dataframe = pd.DataFrame( columns = ["preds_family", "labels_family"]) 
            
                dataframe["preds_family"] = result.predictions[0].argmax(axis=1).squeeze()
                
                
                dataframe["labels_family"] = result.label_ids


                import pickle

                
                pickle.dump({'preds':result.predictions,'labels':result.label_ids}, open(os.path.join(log_dir,'checkpoints',fold,"predictions.pkl"), 'wb'))
                dataframe.to_csv(os.path.join(log_dir,'checkpoints',fold,"predictions.csv"), index=False)


if __name__ == "__main__":
    
    main()
    # pass
    # d= ['macro_accuracy_order', 'micro_accuracy_order', 'macro_accuracy_family', 'micro_accuracy_family']
    # plot_save_loss(r"C:\Users\Auguste Verdier\Desktop\TeleoClassification\outputs\TeleoSplitGenera_300_medium\DNABERT-2-117\multiTaxa", metrics = d)
    