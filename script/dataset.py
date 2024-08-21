
import numpy as np
from torch.utils.data import Dataset


LOW_NUC = 2/40

HIGHT_NUC = 1/20

dict_mutation={'C':['T','G','A'],'T':['C','G','A'],'G':['T','C','A'],'A':['T','G','C']}

    
def mutation(seq,mutation_ratio_min=LOW_NUC,mutation_ration_max=HIGHT_NUC):

    index_mutation=np.random.choice(range(len(seq)),int(len(seq)*np.random.uniform(mutation_ratio_min,mutation_ration_max)))
    return ''.join([n if i not in index_mutation else  np.random.choice(dict_mutation[n]) for i,n in enumerate(seq)])

class AugmentedDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer

        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        augmented_sequence = mutation(sequence)
        encodings = self.tokenizer(augmented_sequence, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
        item = {key: val.squeeze() for key, val in encodings.items()}
        item['labels'] = label
        return item
    

class MultiTaxaDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer

        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        encodings = self.tokenizer(sequence, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
        item = {key: val.squeeze() for key, val in encodings.items()}
        item['labels'] = label
        return item