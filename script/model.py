from torch import nn
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig

class DNAEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name="zhihan1996/DNABERT-2-117M", pretrained= None
    ):
        
        super().__init__()


        self.config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=self.config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
        # if pretrained : 
        #     self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        #     self.model = AutoModel.from_pretrained(pretrained, trust_remote_code=True)
        # else :
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        #     self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # for p in self.model.parameters():
        #     p.requires_grad = trainable

    def forward(self,  input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)[0] #get emmbedding
    

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


    dna_model=DNAEncoder()
    dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"

    inputs = dna_model.tokenizer(dna, return_tensors='pt', return_offsets_mapping=True)

    # Extract the token ids and the offsets
    input_ids = inputs["input_ids"]
    offset_mapping = inputs["offset_mapping"]

    # Print input ids and offset mapping for inspection
    print(f"Input IDs: {input_ids}")
    print(f"Offset Mapping: {offset_mapping}")

    # Check the position of the [CLS] token
    cls_token_id = dna_model.tokenizer.cls_token_id
    cls_position = (input_ids[0] == cls_token_id).nonzero(as_tuple=True)[0].item()

    print(f"[CLS] token position: {cls_position}")

    # Obtain the hidden states
    hidden_states = dna_model.model(input_ids)

    # Print hidden states
    # print(hidden_states)