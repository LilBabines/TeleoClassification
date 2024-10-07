# After CRABS extraction on FISHBASE and NCBI, error treshold is set to 3

import pandas as pd
import numpy as np

# load Teleo from ncbi 12S
teleo_12S= pd.read_csv(r"C:\Users\Auguste Verdier\Desktop\TeleoClassification\data\crabs\12S\teleo_12S_3\output_pga_3.tsv", sep='\t',header=None)
teleo_12S.rename(columns={0:'ID_ncbi', 1:'ID', 2:'kingdom', 3:'phylum', 4:'class', 5:'order', 6:'family', 7:'genus', 8:'species', 9:'sequence'}, inplace=True)

# load Teleo from FishBase 12S
teleo_fb = pd.read_csv(r"C:\Users\Auguste Verdier\Desktop\TeleoClassification\data\crabs\fish_base\teleo_fb_3\output_pga_3.tsv", sep='\t',header=None)
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

teleo_nan.to_csv(r"C:\Users\Auguste Verdier\Desktop\TeleoClassification\data\crabs\teleo_nan_bis.tsv", sep='\t', index=False)
assert len(teleo_nan)==172, f"Wrong number of sequences : {len(teleo_nan)} should be 172"


# all_teleo.to_csv(r"C:\Users\Auguste Verdier\Desktop\TeleoClassification\data\crabs\all_teleo_bis.tsv", sep='\t', index=False)

cleaned_teleo = pd.concat([teleo_nan, all_teleo])
# info_dataframe(cleaned_teleo)

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

info_dataframe(teleo_class_A_C)

teleo_class_A_C.to_csv(r"C:\Users\Auguste Verdier\Desktop\TeleoClassification\data\teleo_class_A_C_bis.tsv", sep='\t', index=False)
