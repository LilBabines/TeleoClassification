
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import os

LOW_NUC = 2/40

HIGHT_NUC = 1/20

MUTATION_DICT={'C':['T','G','A'],'T':['C','G','A'],'G':['T','C','A'],'A':['T','G','C']}

    
def mutation(seq, mutation_ratio_min=LOW_NUC, mutation_ration_max=HIGHT_NUC, mutation_dict=MUTATION_DICT):

    index_mutation=np.random.choice(range(len(seq)),int(len(seq)*np.random.uniform(mutation_ratio_min,mutation_ration_max)))
    return ''.join([n if i not in index_mutation else  np.random.choice(mutation_dict[n]) for i,n in enumerate(seq)])





def load_data(path="Data/TeleoSplitGenera_300_medium/"):
    '''Load the dataset from the path/train.csv, path/test.csv, and path/val.csv
    Args:
        path (str): The path to the dataset
    Returns:
        train (pd.DataFrame): The training dataset
        test (pd.DataFrame): The testing dataset
        val (pd.DataFrame): The validation dataset

    '''
    train = pd.read_csv(os.path.join(path , "train.csv"))
    val = pd.read_csv(os.path.join(path ,  "val.csv"))
    if 'test.csv' in os.listdir(path):
        test = pd.read_csv(os.path.join(path ,  "test.csv"))
    else:
        test = None
    return train, test, val

class AugmentedDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer,transform=mutation):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.transform=transform

        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        augmented_sequence = self.transform(sequence)
        encodings = self.tokenizer(augmented_sequence, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
        item = {key: val.squeeze() for key, val in encodings.items()}
        item['labels'] = label
        return item
def encode_self_supervised_dataset(tokenizer, dir_path):
    train, _, val = load_data(dir_path)
    # Tokenize the sequences
    train_encodings = tokenizer(train['sequence'].tolist(), truncation=True, max_length=512)
    val_encodings = tokenizer(val['sequence'].tolist(), truncation=True, max_length=512)
    train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
           
            
        })

    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
       
    })
    return train_dataset, val_dataset, 


def encode_multiTaxa_dataset(tokenizer, dir_path, dynamic_augmentation=False):
    '''Encode the data using the tokenizer for multi-label classification and return Datasets train/val/test.
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
    train, test, val = load_data(dir_path)
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
    

    assert set(val_order_labels) not in set(train_order_labels), "val order labels not in train order labels"
    assert set(test_order_labels) not in set(train_order_labels), "test order labels not in train order labels"
    assert set(val_family_labels) not in set(train_family_labels), "val family labels not in train family labels"
    assert set(test_family_labels) not in set(train_family_labels), "test family labels not in train family labels"


    
    

    
    # If dynamic augmentation is enabled, apply custom dataset logic
    if False : #dynamic_augmentation:
        pass
        # train_dataset = AugmentedDataset(
        #     train['sequence'].tolist(),
        #     {'order': train_order_labels, 'family': train_family_labels},
        #     tokenizer
        # )
    else:
        train_dataset = Dataset.from_dict({
            'input_ids': train_encodings['input_ids'],
            'attention_mask': train_encodings['attention_mask'],
            'labels': list(zip(train_order_labels,train_family_labels)),
            
        })

    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': list(zip(val_order_labels,val_family_labels)),
    })

    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels':list(zip(test_order_labels,test_family_labels)),
    })

    # Create id-to-label and label-to-id mappings
    id2label_order = {i: label for i, label in enumerate(order_encoder.classes_)}
    label2id_order = {label: i for i, label in enumerate(order_encoder.classes_)}
    id2label_family = {i: label for i, label in enumerate(family_encoder.classes_)}
    label2id_family = {label: i for i, label in enumerate(family_encoder.classes_)}

    return train_dataset, val_dataset, test_dataset, id2label_order, label2id_order, id2label_family, label2id_family

def encode_singleTaxa_dataset(tokenizer, dir_path, dynamic_augmentation=False):
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
    train, test, val = load_data(dir_path)
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
    
    assert set(val_labels) not in set(train_labels), "val order labels not in train order labels"
    assert set(test_labels) not in set(train_labels), "test order labels not in train order labels"
    
    
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