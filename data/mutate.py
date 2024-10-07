import random
from tqdm import tqdm
import pandas as pd

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

for fold in range(1, 7):
    df = pd.read_csv(rf"C:\Users\Auguste Verdier\Desktop\TeleoClassification\data\teleo_clean\fold_{fold}\train.csv")
    df = mutate(df)
    df.to_csv(rf"C:\Users\Auguste Verdier\Desktop\TeleoClassification\data\teleo_clean\fold_{fold}\train_med.csv", index=False)
