# After CRABS extraction on mitofish and NCBI, error treshold is set to 3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import math
import json
import os
import random
from tqdm import tqdm

def pre_process(mitophish, ncbi):
    '''
    Pre-process the data extracted from FishBase and NCBI
        :param mitophish: path to the data extracted from FishBase with CRABS ( separated by `tab`, without header)
        :param ncbi: path to the data extracted from NCBI with CRABS ( separated by `tab`, without header)
    '''
    # load Teleo from ncbi 12S
    teleo_12S= pd.read_csv(ncbi, sep='\t',header=None)
    teleo_12S.rename(columns={0:'ID_ncbi', 1:'ID', 2:'kingdom', 3:'phylum', 4:'class', 5:'order', 6:'family', 7:'genus', 8:'species', 9:'sequence'}, inplace=True)

    # load Teleo from FishBase 12S
    teleo_fb = pd.read_csv(mitophish, sep='\t',header=None)
    teleo_fb.rename(columns={0:'ID_ncbi', 1:'ID', 2:'kingdom', 3:'phylum', 4:'class', 5:'order', 6:'family', 7:'genus', 8:'species', 9:'sequence'}, inplace=True)

    # Filter the teleo_12S dataframe to keep only the classes present in the teleo_fb dataframe ie. fish classes
    fish_12S_teleo = teleo_12S[teleo_12S['class'].isin(teleo_fb['class'].unique())]

    # Concatenate the two dataframes
    all_teleo = pd.concat([fish_12S_teleo, teleo_fb])

    # Fill nan ( add 170 sequences)

    teleo_nan = all_teleo[all_teleo['class'].isna()]
    all_teleo = all_teleo.dropna(subset=['class'])

    teleo_nan.loc[teleo_nan['order'].isin(['Coelacanthiformes', 'Ceratodontiformes']), 'class'] = 'Sarcopterygii'

    teleo_nan = teleo_nan[~teleo_nan['order'].isin(['Testudines', 'Crocodylia', 'Diplura'])]

    teleo_nan = teleo_nan[teleo_nan['ID_ncbi'] != "KM078797.1.1318.2292"]

    teleo_nan.loc[teleo_nan['ID_ncbi'].isin(['AB626856.1.70.1026','AB626856']), ['kingdom', 'phylum', 'class', 'order', 'family', 'genus','species']] = ['Eukaryota', 'Chordata', 'Actinopteri', 'Cypriniformes', 'Leuciscidae', 'Pseudaspius','Pseudaspius_sachalinensis']

    teleo_nan.loc[teleo_nan['ID_ncbi'].isin(['AP011270.1.70.1026','AP011270']), ['kingdom', 'phylum', 'class', 'order', 'family', 'genus','species']] = ['Eukaryota', 'Chordata', 'Actinopteri', 'Cypriniformes', 'Leuciscidae', 'Pseudaspius','Pseudaspius_sachalinensis']

    assert len(teleo_nan)==172, f"Wrong number of sequences : {len(teleo_nan)} should be 172"

    cleaned_teleo = pd.concat([teleo_nan, all_teleo])

    # Filter sequence lenght < 20
    cleaned_teleo_correct_len = cleaned_teleo[cleaned_teleo['sequence'].str.len() >= 20]

    # Filter sequience contains N
    cleaned_teleo_no_N = cleaned_teleo_correct_len[cleaned_teleo_correct_len['sequence'].str.contains('N') == False]

    # Fill NaN
    dic_family_to_order = {"Scatophagidae": 'Perciformes',
                        'Sillaginidae' : 'Perciformes',
                        'Plesiopidae'  : 'Perciformes',
                        'Pomacanthidae' : 'Perciformes',
                        "Sciaenidae": "Acanthuriformes",
                        "Ambassidae"  : "Perciformes",
                        "Pseudochromidae"  : "Perciformes",
                        "Polycentridae" : "Perciformes",
                        "Opistognathidae"  : "Perciformes",
                        "Toxotidae" : "Perciformes",
                        'Pristiophoridae' : 'Pristiophoriformes',
                        'Platyrhinidae' : 'Torpediniformes',
                        'Emmelichthyidae' : 'Acanthuriformes',
                        'Pomacentridae':  "Perciformes",
                        'Embiotocidae' :'Perciformes',
                        'Siganidae': 'Perciformes',
                        'Squatinidae' : 'Squatiniformes',
                        "Centropomidae" : "Perciformes" ,
                        "Malacanthidae" : "Perciformes" ,
                        'Polynemidae' :  'Perciformes' , 
                        'Moronidae' :  'Perciformes' ,
                        'Menidae' :  'Perciformes' ,
                        "Lactariidae" : "Perciformes",
                        "Sphyraenidae" : "Perciformes",
                        'Callanthiidae' : 'Perciformes',
                        "Monodactylidae" : "Perciformes"}

    dic_order_to_class = {'Coelacanthiformes': "Sarcopterygii",
                        'Ceratodontiformes' : 'Sarcopterygii'}

    dic_genus_to_family = {'Percalates': 'Percichthyidae',
                        'Paedocypris': 'Cyprinidae',
                        'Bembrops' : 'Percophidae',
                        'Conorhynchos' : 'Pimelodidae',
                        'Lepidogalaxias' :'Lepidogalaxiidae',
                        }
    def fill_nan(df):
        df = df.copy()
        for key,item in dic_family_to_order.items():
            idx = df[df['family']==key]
            df.loc[idx.index, 'order'] = item
        for key,item in dic_order_to_class.items():
            idx = df[df['order']==key]
            df.loc[idx.index, 'class'] = item
        for key,item in dic_genus_to_family.items():
            idx = df[df['genus']==key]
            df.loc[idx.index, 'family'] = item

        df =df.dropna()
        
        return df


    teleo = fill_nan(cleaned_teleo_no_N)

    # Keep only Actinopteri and Chondrichthyes
    teleo_class_A_C = teleo[(teleo['class'] == 'Actinopteri') | (teleo['class'] == 'Chondrichthyes')]

    teleo_class_A_C.to_csv("teleo_clean.tsv", sep='\t', index=False)

    return teleo_class_A_C 

def fold_6_data(data_path, n_splits=6):
    '''
    Fold the data into 6 folds with genus as the stratification
        :param data_path: path to the pre-processed data (teleo_clean.tsv, separated by `tab`)
    '''
    # Split the data into train and test set
    df = pd.read_csv(data_path, sep='\t')
    

    for k in range(n_splits):
        # fill the fold column with -1
        df[f'fold_{k}'] = "none"


    families = df['family'].unique()

    for family in families:

        family_data = df[df['family'] == family]
        genus_labels = family_data['genus'].unique()
        genus_number = family_data['genus'].nunique()

        # exclude family with only one genus (cant be split)
        if genus_number ==1:
            continue
        
        # if the number of genus is smaller than the number of splits
        elif genus_number < n_splits :
            skf = KFold(n_splits=genus_number)

        else :
            skf = KFold(n_splits=n_splits)

        
        # shuffle the fold assignment
        random_fold_assigment = np.arange(min(genus_number, n_splits))
        np.random.shuffle(random_fold_assigment)

        # partition data at genus level
        for fold, (train_index, val_index) in enumerate(skf.split(genus_labels)):
            
            # get the genus labels for the current fold
            val_genus = genus_labels[val_index]
            train_genus = genus_labels[train_index]

            # select validation data
            val_data = df[df['genus'].isin(val_genus)]

            if len(val_data)>1:
                # if there are more than 1 sample in the validation set, 50/50 split it into validation and test
                val_index , test_index = train_test_split(val_data.index, test_size=0.5)
            else:
                # if there is only one sample in the validation set, use it as test
                val_index = []
                test_index = val_data.index

            # fill the fold column
            df.loc[val_index, f'fold_{random_fold_assigment[fold]}'] = 'val'
            df.loc[test_index, f'fold_{random_fold_assigment[fold]}'] = 'test'
            df.loc[df['genus'].isin(train_genus), f'fold_{random_fold_assigment[fold]}'] = 'train'

        # fill the other fold if genus_number < n_splits
        if fold < n_splits-1:
            
            a=np.arange(fold+1)
            np.random.shuffle(a)
            dulicate = np.tile(a, math.ceil((n_splits-fold-1)/(fold+1)))[:n_splits-fold-1]
            for i,f in enumerate(range(fold + 1, n_splits)):
                
                # duplicate the fold assignment based on `duplicate` index
                df.loc[family_data.index, f'fold_{f}'] = df.loc[family_data.index, f'fold_{dulicate[i]}'].copy()
    # save the data
    df.to_csv("teleo_clean_fold.tsv", sep='\t', index=False)

    return df

def get_repartition(data_path):
    '''
    Get the repartition of the data in each fold
        :param data_path: path to the pre-processed data with fold (teleo_clean_fold.tsv, separated by `tab`)
    '''
    df = pd.read_csv(data_path, sep='\t')

    repartition = {}

    for i in range(6):
        print("-------------------")
        print(f"Fold {i+1}")

        val =df[df[f'fold_{i}'] == 'val']
        train =df[df[f'fold_{i}'] == 'train']
        test =df[df[f'fold_{i}'] == 'test']
        val_test = pd.concat([val,test])

        print(f"Train genus ratio : {print(train['genus'].nunique() / (train['genus'].nunique() + val_test['genus'].nunique() ))}")
        print(f'Train samples ratio : {train.shape[0] /(val.shape[0] + train.shape[0] + test.shape[0])}')
        print(f"Val samples ratio : {val.shape[0] /(val.shape[0] + train.shape[0] + test.shape[0])}")
        print(f"Test samples ratio : {test.shape[0] /(val.shape[0] + train.shape[0] + test.shape[0])}")

        assert set(val_test['genus']).isdisjoint(set(train['genus']))
        repartition[i] = df[f'fold_{i}'].value_counts().to_dict()

    return repartition

def build_file(data_path):
    '''
    Build the file for the training
        :param data_path: path to the pre-processed data with fold (teleo_clean_fold.tsv, separated by `tab`)
    '''
    df_with_folds = pd.read_csv(data_path, sep='\t')
    df = df_with_folds.drop_duplicates(subset=['sequence','genus'], keep='first')

    os.makedirs("data/teleo_clean", exist_ok=True)
    for i in range(6):

        os.makedirs(f"/data/teleo_clean/fold_{i+1}", exist_ok=True)
        
        train = df_with_folds[df_with_folds[f'fold_{i}'] == 'train']
        val = df_with_folds[df_with_folds[f'fold_{i}'] == 'val']
        test = df_with_folds[df_with_folds[f'fold_{i}'] == 'test']

        #save genus for validation and test
        json_data = pd.concat([val,test])['genus'].unique().tolist()
        json.dump(json_data, open(rf"data/teleo_clean/fold_{i+1}/test_genus.json", 'w'))

        # We want to classify the sequences by family, so we need to remove duplicates at the family level
        train = train.drop_duplicates(subset=['sequence','family'], keep='first')
        val = val.drop_duplicates(subset=['sequence','family'], keep='first')
        test = test.drop_duplicates(subset=['sequence','family'], keep='first')

        val[['sequence', 'order' , 'family', 'genus','species']].to_csv(f"data/teleo_clean/fold_{i+1}/val.csv", sep=',', index=False, header=True)
        train[['sequence', 'order' , 'family', 'genus', 'species']].to_csv(rf"data/teleo_clean/fold_{i+1}/train.csv", sep=',', index=False, header=True)
        test[['sequence', 'order' , 'family', 'genus', 'species']].to_csv(rf"data/teleo_clean/fold_{i+1}/test.csv", sep=',', index=False, header=True)

def mutate_dna_sequence(sequence, mutation_probability):
    mutated_sequence = ""
    
    for base in sequence:
        if random.random() < mutation_probability:
            mutated_base = random.choice(['A', 'T', 'C', 'G'])
            while mutated_base == base:
                mutated_base = random.choice(['A', 'T', 'C', 'G'])
            mutated_sequence += mutated_base
        else:
            mutated_sequence += base
    
    return mutated_sequence

def split_and_mutate_sequence(sequence):
    # Calculate the lengths of each part
    length_1 = int(len(sequence) * 0.06)
    length_2 = int(len(sequence) * 0.62)
    length_3 = int(len(sequence) * 0.29)
    
    # Split the sequence into three parts
    part_1 = sequence[:length_1]
    part_2 = sequence[length_1:length_1+length_2]
    part_3 = sequence[length_1+length_2:]
    
    # Apply mutate_dna_sequence to each part with respective probabilities
    mutated_part_1 = mutate_dna_sequence(part_1, 0.024)
    mutated_part_2 = mutate_dna_sequence(part_2, 0.24)
    mutated_part_3 = mutate_dna_sequence(part_3, 0.008)
    
    # Concatenate the mutated parts into a new sequence
    mutated_sequence = mutated_part_1 + mutated_part_2 + mutated_part_3
    
    # Return the mutated sequence
    return mutated_sequence

def mutate(df):
    # Calculate the value count of the "family" column
    family_counts = df['family'].value_counts()

    # Get the families with a count below 200
    families_below_200 = family_counts[family_counts < 300].index

    # Duplicate rows for families with a count below 200
    for family in tqdm(families_below_200):
        count = family_counts[family]
        if count == 1:
            # Duplicate the only row 199 times
            duplicated_rows = pd.concat([df[df['family'] == family]] * 299, ignore_index=True)
        else:
            # Duplicate random members of the family to reach a count of 200
            duplicated_rows = df[df['family'] == family].sample(n=300-count, replace=True)
        
        # Apply mutations to each duplicated row's sequence
        duplicated_rows['sequence'] = duplicated_rows['sequence'].apply(lambda seq: split_and_mutate_sequence(seq))
        
        # Change the species to "Synthetic" for the duplicated rows
        duplicated_rows['species'] = 'Synthetic'
        
        df = pd.concat([df, duplicated_rows], ignore_index=True)

    return df

def mutate_data_fold(data_path = "data/teleo_clean"):
    '''
    Mutate the data
        :param data_path: folds location (data/teleo_clean)
    '''
    
    for fold in range(1, 7):
        df = pd.read_csv(data_path +f"/fold_{fold}/train.csv")
        df = mutate(df)
        df.to_csv(data_path +f"/fold_{fold}/train_med_augment.csv", index=False)
