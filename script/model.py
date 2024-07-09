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
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=TEMPERATURE,
        image_embedding=768,
        text_embedding=768,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


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
 
    
    

