
from tokenizer import KmerTokenizer
from model import DNABERTWithDropout
from dataset import AugmentedDataset
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import contextlib

import evaluate
f1_metric = evaluate.load("f1")


from transformers import BertConfig, AutoTokenizer
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer

from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

from transformers import AutoTokenizer,PreTrainedTokenizerFast




@contextlib.contextmanager
def suppress_transformers_logging():
    logger = logging.getLogger("transformers")
    previous_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous_level)

    
def plot_save_loss(result_path):
    '''Plot and save the loss and f1 score from the result_path/trainer_state.json file
    Args:
        result_path (str): The path to the result folder
    '''
    with open(os.path.join(result_path,'trainer_state.json'), 'r') as f:
        data = json.load(f)
        log=data['log_history']
        #print(type(log))
    x_train_loss=[]
    y_train_loss=[]

    x_val_loss=[]
    y_val_loss=[]

    y_f1=[]

    for item in log:
        
        if 'loss' in item.keys():

            x_train_loss.append(item['epoch'])

            y_train_loss.append(item['loss']) 
            
        else :

            x_val_loss.append(item['epoch'])

            y_val_loss.append(item['eval_loss']) 
            y_f1.append(item['eval_f1'])



    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Train Val Loss, FISH Teleo 300 Medium 3 Mer")
    plt.plot(x_train_loss,y_train_loss,label='Train Loss')
    plt.plot(x_val_loss,y_val_loss,label='Val Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'Images/LOSS_{result_path.split("/")[-1]}.png')
    plt.savefig(f'{result_path}/LOSS.png')
    plt.show()


    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.title("Val Performance, FISH Teleo 300 Medium 3 Mer")
    #plt.plot(x_train_loss,y_train_loss,label='Train Loss')
    plt.plot(x_val_loss,y_f1)
    plt.grid(True)
    plt.savefig(f'{result_path}/PERF.png')
    plt.savefig(f'Images/PERF_{result_path.split("/")[-1]}.png')
    plt.show()

def load_dataset(path="Data/TeleoSplit/"):
    '''Load the dataset from the path/train.csv, path/test.csv, and path/val.csv
    Args:
        path (str): The path to the dataset
    Returns:
        train (pd.DataFrame): The training dataset
        test (pd.DataFrame): The testing dataset
        val (pd.DataFrame): The validation dataset

    '''
    train = pd.read_csv(path + "train.csv")
    test = pd.read_csv(path + "test.csv")
    val = pd.read_csv(path + "val.csv")
    return train, test, val

def load_tokenizer(name="zhihan1996/DNABERT-2-117M"):
    '''Load the tokenizer from the model name or the kmer size
    Args:
        name (str or int): The model name or the kmer size
    Returns:
        tokenizer (PreTrainedTokenizerFast): The tokenizer
    '''
    if isinstance(name, str):

        if name=="bpe":

            tokenizer = PreTrainedTokenizerFast(tokenizer_file="Model/tokenizer/teleo_4096.json")

            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            tokenizer.add_special_tokens({'sep_token': '[SEP]'})
            tokenizer.add_special_tokens({'cls_token' : '[CLS]'})
            tokenizer.add_special_tokens({'unk_token' : '[UNK]'})

        else :
            tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

     
    elif isinstance(name, int):
        tokenizer = KmerTokenizer(name)

       
    return tokenizer
def encode_multiTaxa_data(tokenizer, train, val, test, dynamic_augmentation=False):
    '''Encode the data using the tokenizer for multi-label classification.
    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer
        train (pd.DataFrame): The training dataset
        val (pd.DataFrame): The validation dataset
        test (pd.DataFrame): The testing dataset
    Returns:
        train_dataset (Dataset): The training dataset
        val_dataset (Dataset): The validation dataset
        test_dataset (Dataset): The testing dataset
        id2label_order (dict): The id to order label mapping
        label2id_order (dict): The order label to id mapping
        id2label_family (dict): The id to family label mapping
        label2id_family (dict): The family label to id mapping
    '''

    # Tokenize the sequences
    train_encodings = tokenizer(train['sequence'].tolist(), truncation=True, max_length=512)
    val_encodings = tokenizer(val['sequence'].tolist(), truncation=True, max_length=512)
    test_encodings = tokenizer(test['sequence'].tolist(), truncation=True, max_length=512)

    # Initialize the label encoders
    order_encoder = LabelEncoder()
    family_encoder = LabelEncoder()

    # Fit the label encoders on the combined data
    order_encoder.fit(pd.concat([train['order'], val['order'], test['order']]))
    family_encoder.fit(pd.concat([train['family'], val['family'], test['family']]))

    # Encode the labels
    train_order_labels = order_encoder.transform(train['order'])
    val_order_labels = order_encoder.transform(val['order'])
    test_order_labels = order_encoder.transform(test['order'])

    train_family_labels = family_encoder.transform(train['family'])
    val_family_labels = family_encoder.transform(val['family'])
    test_family_labels = family_encoder.transform(test['family'])

    # If dynamic augmentation is enabled, apply custom dataset logic
    if dynamic_augmentation:
        train_dataset = AugmentedDataset(
            train['sequence'].tolist(),
            {'order': train_order_labels, 'family': train_family_labels},
            tokenizer
        )
    else:
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'order_labels': train_order_labels,
            'family_labels': train_family_labels
        })

    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'order_labels': val_order_labels,
        'family_labels': val_family_labels
    })

    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'order_labels': test_order_labels,
        'family_labels': test_family_labels
    })

    # Create id-to-label and label-to-id mappings
    id2label_order = {i: label for i, label in enumerate(order_encoder.classes_)}
    label2id_order = {label: i for i, label in enumerate(order_encoder.classes_)}
    id2label_family = {i: label for i, label in enumerate(family_encoder.classes_)}
    label2id_family = {label: i for i, label in enumerate(family_encoder.classes_)}

    return train_dataset, val_dataset, test_dataset, id2label_order, label2id_order, id2label_family, label2id_family

def encode_data(tokenizer, train, val, test, dynamic_augmentation=False):
    '''Encode the data using the tokenizer
    Args:
        tokenizer (PreTrainedTokenizerFast): The tokenizer
        train (pd.DataFrame): The training dataset
        val (pd.DataFrame): The validation dataset
        test (pd.DataFrame): The testing dataset
    Returns:
        train_dataset (Dataset): The training dataset
        val_dataset (Dataset): The validation dataset
        test_dataset (Dataset): The testing dataset
        id2label (dict): The id to label mapping
        label2id (dict): The label to id mapping
    '''
    if not dynamic_augmentation:

        train_encodings = tokenizer(train['sequence'].tolist(), truncation=True, max_length=512)
    val_encodings = tokenizer(val['sequence'].tolist(), truncation=True, max_length=512)
    test_encodings = tokenizer(test['sequence'].tolist(), truncation=True, max_length=512)

    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Fit the label encoder on the combined data
    label_encoder.fit(pd.concat([train['family'], val['family'], test['family'] ]  ))

    # Encode the labels
    train_labels = label_encoder.transform(train['family'])
    val_labels = label_encoder.transform(val['family'])
    test_labels = label_encoder.transform(test['family'])
    
    if dynamic_augmentation:

        train_dataset = AugmentedDataset(
            train['sequence'].tolist(),
            train_labels,
            tokenizer
        )
    else:
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': train_labels
        })
    

    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': val_labels
    })

    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels':test_labels
    })
    id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
    label2id = {label: i for i, label in enumerate(label_encoder.classes_)}

    return train_dataset, val_dataset, test_dataset, id2label, label2id

def load_model(name, vocab_size, local=False, id2label=None, label2id=None,dropout=False):

    # model_path_save=r"C:\Users\Auguste Verdier\Desktop\ADNe\BouillaClip\Model\genera_300_medium_3_mer\checkpoint-85335"
    if local:
        assert os.path.exists(name), "The model path does not exist at the specified location, but local flag is set to True"
        assert os.path.exists(os.path.join(name,"config.json")), "The model path does not contain a config.json file"
        config_path = os.path.join(name,"config.json")
    else :
        config_path = name
        
    config = BertConfig.from_pretrained(config_path, 
                                        num_labels=len(id2label), 
                                        max_position_embeddings=514,
                                        id2label=id2label,
                                        label2id=label2id)

    if dropout:

        with suppress_transformers_logging():
            model = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True, ignore_mismatched_sizes=True, config=config)

        model = DNABERTWithDropout.from_pretrained(
            "zhihan1996/DNABERT-2-117M", 
            trust_remote_code=True, 
            ignore_mismatched_sizes=True, 
            config=config
        )

    else:
        model = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True, ignore_mismatched_sizes=True, config=config)

    model.id2label = id2label
    model.label2id = label2id
    model.resize_token_embeddings(vocab_size)
    return model

def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids


    if isinstance(predictions, tuple):
        predictions = predictions[0]
    # Ensure predictions are 2D
    if predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=1)
    elif predictions.ndim > 2:
        raise ValueError(f"Unexpected predictions shape: {predictions.shape}")
    
    return f1_metric.compute(predictions=predictions, references=labels, average='macro')  

def training_argument(output_path, learning_rate=1e-5, per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=20, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True, push_to_hub=False):
    arg=TrainingArguments(
        output_dir=output_path,
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
    return arg

def define_trainer(model, tokenizer, train_dataset, val_dataset, training_args):

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )



    return trainer


