from torch import nn
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
from itertools import product
import tokenizer


class DNAEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="zhihan1996/DNABERT-2-117M", kmer = None
    ):
        
        super().__init__()


        self.config = BertConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, config=self.config)
        


        if kmer :
            
            self.tokenizer = tokenizer.KmerTokenizer(k=kmer)
            
        else :
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.target_token_idx = 0

    def forward(self,  input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)[0:self.target_token_idx,:] #get emmbedding
    

class TaxEncoder(nn.Module):
    def __init__(self, model_name, pretrained):
        super().__init__()

        if pretrained:
            self.model = 0
        else:
            self.model = 0
            
        # for p in self.model.parameters():
        #     p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


if __name__=='__main__':

    dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"

    dna_model=DNAEncoder(kmer=4,model_name="zhihan1996/DNABERT-S") #

    inputs = dna_model.tokenizer(dna, return_tensors='pt')


    input_ids = inputs["input_ids"]
    
    cls_token_id = dna_model.tokenizer.cls_token_id

    hidden_states = dna_model.model(input_ids)

    decode_input = dna_model.tokenizer.decode(input_ids[0])
    print(f"Input sequence: {decode_input}")

    print(hidden_states[0].shape)  # (batch_size, sequence_length, hidden_size)