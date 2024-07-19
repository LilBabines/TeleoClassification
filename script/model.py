from torch import nn
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
from itertools import product
import tokenizer
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch
from torch import nn

DIM_emmbedding = 768
DROPOUT = 0.1
TEMPERATURE = 0.07

print(os.getcwd())



from transformers import BertForSequenceClassification, AutoModelForSequenceClassification

class DNABERTWithDropout(BertForSequenceClassification):
    ''' 
    A BERT model with an additional dropout layer applied to the input embeddings

    '''
    def __init__(self, config, dropout_prob=0.3):
        
        super().__init__(config)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # Apply dropout to the input embeddings
        if input_ids is not None:
            inputs_embeds = self.bert.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
            inputs_embeds = self.dropout(inputs_embeds)
        
        # Pass the modified embeddings to the original forward method
        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    

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
        return self.model(input_ids=input_ids, attention_mask=attention_mask)[0][0,self.target_token_idx,:] 
    

class TaxEncoder(nn.Module):
    def __init__(self):

        super().__init__()

        # Initialize a GPT-2 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        special_tokens = self.tokenizer.special_tokens_map.values()

        new_vocab = {token: i for i, token in enumerate(special_tokens)}

        self.tokenizer.vocab = new_vocab
        self.tokenizer.encoder = new_vocab
        self.tokenizer.decoder = {i: token for token, i in new_vocab.items()}
        
        with open('token_taxa_list.txt') as f:
            token_list = f.read().splitlines()

        self.tokenizer.add_tokens(token_list)
        self.tokenizer.add_tokens(' ')

        # Initialize a GPT-2 model
        self.model = GPT2LMHeadModel.from_pretrained('./TaxaGPT2/', config='./TaxaGPT2/config.json')

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)[0][:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=DIM_emmbedding,
        dropout=DROPOUT
    ):
        super().__init__()

        self.projection_order = nn.Linear(embedding_dim, projection_dim)
        self.fc_order = nn.Linear(projection_dim, projection_dim)

        self.projection_family = nn.Linear(embedding_dim, projection_dim)
        self.fc_family = nn.Linear(projection_dim, projection_dim)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

        
    
    def forward(self, x):

        projected_order = self.projection_order(x)
        x_order = self.gelu(x_order)
        x_order = self.fc_order(x_order)
        x_order = self.dropout(x_order)
        x_order = x_order + projected_order
        x_order = self.layer_norm(x_order)

        projected_family = self.projection_family(x)
        x_family = self.gelu(x_family)
        x_family = self.fc_family(x_family)
        x_family = self.dropout(x_family)
        x_family = x_family + projected_family
        x_family = self.layer_norm(x_family)

        return x_order, x_family


class BouillaClip(nn.Module):
    def __init__(
        self,
        temperature=TEMPERATURE,
        dna_embedding=768,
        taxa_embedding=768,
    ):
        super().__init__()
        self.dna_encoder = DNAEncoder()
        self.taxa_encoder = TaxEncoder()

        self.dna_projection = ProjectionHead(embedding_dim=dna_embedding)
        self.taxa_projection = ProjectionHead(embedding_dim=taxa_embedding)


        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        dna_features = self.dna_encoder(batch["dna_input_ids"], batch["dna_attention_mask"])
        taxa_features = self.taxa_encoder(
            input_ids=batch["taxa_input_ids"], attention_mask=batch["taxa_attention_mask"]
        )

        # Getting Image and Text Embeddings (with same dimension)
        dna_embedding_order, dna_embedding_family = self.dna_projection(dna_features)
        taxa_embedding_order, taxa_embedding_family = self.taxa_projection(taxa_features)

        # Calculating the Loss
        loss_order = contrastive_loss(dna_embedding_order, taxa_embedding_order, self.temperature)
        loss_family = contrastive_loss(dna_embedding_family, taxa_embedding_family, self.temperature)

        return loss_order + loss_family


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
if __name__=='__main__':


    # ----------------- DNA Encoder -----------------
    dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"

    dna_model=DNAEncoder(kmer=4,model_name="zhihan1996/DNABERT-S") #

    inputs = dna_model.tokenizer(dna, return_tensors='pt')


    input_ids = inputs["input_ids"]
    atention_attention_mask = inputs["attention_mask"]

    cls_token_id = dna_model.tokenizer.cls_token_id

    hidden_states = dna_model(input_ids, atention_attention_mask)

    decode_input = dna_model.tokenizer.decode(input_ids[0])
    print(f"Input sequence: {decode_input}")

    print(hidden_states.shape)  # (batch_size, sequence_length, hidden_size)

    # ----------------- Taxa Encoder -----------------
    taxa_model = TaxEncoder()
    taxa_model.model.eval()

    # Set the word you want to generate text for
    input_word = "Chordata Actinopteri Zeiformes "

    # Tokenize the input word
    input_ids = taxa_model.tokenizer.encode(input_word, return_tensors="pt")

    # Move the input_ids tensor to the same device as the model
    input_ids = input_ids.to(taxa_model.model.device)

    # Generate text based on the input word
    output = taxa_model.model.generate(input_ids) #, max_length=2000, num_return_sequences=1

    # Decode the generated text
    generated_text = taxa_model.tokenizer.decode(output[0], skip_special_tokens=True)

    # Print the generated text
    print(generated_text)
 
    
    

