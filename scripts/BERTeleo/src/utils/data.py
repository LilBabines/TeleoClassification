from ete3 import NCBITaxa
import pandas as pd

class TaxidLineage:

    def __init__(self):
        self.ncbi = NCBITaxa()
       
    def _get_d_rank(self, d, rank):
        if (rank not in d.values()):
            return (None, 'unknown')
        taxid = [k for k, v in d.items() if v == rank]
        name = self.ncbi.translate_to_names(taxid)[0]
        return name if isinstance(name, str) else 'unknown' #taxid[0], 

    def get_ranks(self, taxid, ranks=['order', 'family']):
        d = self.ncbi.get_rank(self.ncbi.get_lineage(taxid))
        print(d)
        if len(ranks) == 1 or isinstance(ranks, str):
            return self._get_d_rank(d, ranks[0])
        return [self._get_d_rank(d, r) for r in ranks] 
    
    def get_id_from_name(self, name):

        # if name =='Gonostomatidae' :
        #     return 48439 # this is a special case
        # elif name == 'Chilodontidae':
        #     return 42619 # this is a special case

        return self.ncbi.get_name_translator([name])[name][0]




def load_data(path="Data/TeleoSplitSlack/"):

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

def clean_taxa_dataframe(df, taxid_lineage):
    '''Clean the dataframe by adding the order and family columns from family_taxid
    Args:
        df (pd.DataFrame): The dataframe to clean
        taxid_lineage (TaxidLineage): The TaxidLineage object to use
    Returns:
        df (pd.DataFrame): The cleaned dataframe

    '''
    df_clean = pd.DataFrame(columns=["order", "family", "taxid"])

    df_clean['sequence'] = df['sequence']
    df_clean["taxid"] = df["family"].apply(taxid_lineage.get_id_from_name)
    df_clean["order"] = df_clean["taxid"].apply(lambda x: taxid_lineage.get_ranks(x, ranks=['order']))
    df_clean["family"] = df_clean["taxid"].apply(lambda x: taxid_lineage.get_ranks(x, ranks=['family']))    

    df_dif = df_clean[df_clean["order"] != df["order"]]
    
    print(df_dif.sample(20))
    return df_clean

if "__main__" == __name__:
    taxid_lineage = TaxidLineage()
    family = "Pomacentridae"
    taxID= 30863

    print(taxid_lineage.get_ranks(taxID, ranks=['order', 'family']))
       
    # print(train_clean.head())
    