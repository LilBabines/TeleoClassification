from torch import nn
import torch

import os
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModel, AutoModel, BertConfig, BertForMaskedLM

from transformers import AutoModelForSequenceClassification
from datasets import load_dataset


#BarCodeBERT : https://github.com/Kari-Genomics-Lab/BarcodeBERT/blob/main/scripts/BarcodeBERT/Fine-tuning.py
# Paper : https://arxiv.org/abs/2311.02401
class BarcodeBert_model(nn.Module):
    def __init__(self, checkpoint, num_labels, vocab_size):
        super(BarcodeBert_model, self).__init__()
        self.num_labels = num_labels
        # Load Model with given checkpoint
        self.model = BertForMaskedLM(BertConfig(vocab_size=int(vocab_size), output_hidden_states=True))
        self.model.load_state_dict(torch.load(checkpoint, map_location="cuda:0", weights_only=True), strict=False )
        self.classifier = nn.Linear(768, self.num_labels)

    def forward(self, input_ids=None, labels=None, attention_mask=None):
        # Getting the embedding
        outputs = self.model(input_ids=input_ids,attention_mask=attention_mask)
        embeddings = outputs.hidden_states[-1]
        GAP_embeddings = embeddings.mean(1)
        # calculate losses
        logits = self.classifier(GAP_embeddings.view(-1, 768))
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

        # return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states)

class HierachicalLoss(nn.Module):
    def __init__(self, num_orders, num_families, compatibility_matrix, lambda_penalty=0.1, loss_fn_order=None, loss_fn_family=None):
        super(HierachicalLoss, self).__init__()

        # Loss functions for order and family
        self.loss_fn_order = loss_fn_order if loss_fn_order else nn.CrossEntropyLoss()
        self.loss_fn_family = loss_fn_family if loss_fn_family else nn.CrossEntropyLoss()
        
        # Compatibility matrix for hierarchical relationships
        self.compatibility_matrix = compatibility_matrix  # Shape: (num_orders, num_families)
        self.lambda_penalty = lambda_penalty  # Regularization strength for hierarchical penalty

        # Ensure compatibility matrix is the correct size and is a tensor
        assert compatibility_matrix.shape == (num_orders, num_families)
        self.compatibility_matrix = torch.tensor(compatibility_matrix, dtype=torch.float32)

    def forward(self, logits_order, logits_family, labels_order, labels_family):
        # Compute order loss
        loss_order = self.loss_fn_order(logits_order, labels_order)
        
        # Compute family loss
        loss_family = self.loss_fn_family(logits_family, labels_family)

        # Total loss is the sum of the two losses
        total_loss = loss_order + loss_family
        
        # Add hierarchical penalty based on compatibility
        with torch.no_grad():
            # Get the predicted order and family
            pred_order = torch.argmax(logits_order, dim=-1)  # Shape: (batch_size,)
            pred_family = torch.argmax(logits_family, dim=-1)  # Shape: (batch_size,)

            # Check if both order and family are incorrect
            both_incorrect = (pred_order != labels_order) & (pred_family != labels_family)

            # Retrieve compatibility between predicted orders and families
            compatibility_scores = self.compatibility_matrix[pred_order, pred_family]  # Shape: (batch_size,)

            # Penalize if the prediction is incorrect and incompatible
            hierarchical_penalty = both_incorrect.float() * (1 - compatibility_scores)

        # Add the hierarchical penalty to the total loss
        total_loss += self.lambda_penalty * hierarchical_penalty.mean()

        return total_loss


class MultiTaxaClassification(nn.Module):
    def __init__(self, num_labels_order = 72, num_labels_family = 303, vocab_size = None,**bert_kwargs ):
        super(MultiTaxaClassification,self).__init__()
        
        self.num_labels = (num_labels_order, num_labels_family)
        self.problem_type = "multi_label_classification"
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M",vocab_size= vocab_size,**bert_kwargs )
        self.bert = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True,config=config, ignore_mismatched_sizes=True)
        self.bert.resize_token_embeddings(vocab_size)
        
        hidden_size = self.bert.config.hidden_size
        classifier_dropout = 0.1

        self.dropout = nn.Dropout(classifier_dropout)
        

        self.classifier_order = nn.Linear(hidden_size, self.num_labels[0])
        self.classifier_family = nn.Linear(hidden_size + self.num_labels[0] , self.num_labels[1]) # Concatenate the order logits to the family logits
        # self.classifier_family = nn.Linear(hidden_size, self.num_labels[1]) 

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True,
    ):
        
        

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            )
        
        
        # Use the [CLS] token's output (first token of the sequence)
        pooled_output = outputs[0][:, 0, :]  # Shape: [batch_size, hidden_size]

        pooled_output = self.dropout(pooled_output)
        
        logits_order = self.classifier_order(pooled_output)
        logits_family = self.classifier_family(torch.cat((pooled_output, logits_order), dim=1))
        logits = (logits_order, logits_family)
        
        
        loss = None
        
        if labels is not None:
            labels_order, labels_family = labels[:,0], labels[:,1]

            loss_fct = nn.CrossEntropyLoss()
            loss_order = loss_fct(logits_order, labels_order)
            loss_family = loss_fct(logits_family, labels_family)
            loss = loss_order + loss_family
     
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        # return {"loss":loss, "logits":logits}
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )


class DNABERTWithDropout(AutoModelForSequenceClassification):
    def __init__(self, config, dropout_prob=0.3):
        super().__init__(config)
        self.dropout_input_layer = nn.Dropout(dropout_prob)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # Apply dropout to the input embeddings
        if input_ids is not None:
            inputs_embeds = self.bert.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
            inputs_embeds = self.dropout_input_layer(inputs_embeds)
        
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




def load_bert_model(name, vocab_size, local=False, id2label=None, label2id=None):

    # model_path_save=r"C:\Users\Auguste Verdier\Desktop\ADNe\BouillaClip\Model\genera_300_medium_3_mer\checkpoint-85335"
    if local:
        assert os.path.exists(name), "The model path does not exist at the specified location, but local flag is set to True"
        assert os.path.exists(os.path.join(name,"config.json")), "The model path does not contain a config.json file"
        config_path = os.path.join(name,"config.json")
    else :
        config_path = name
        
    config = BertConfig.from_pretrained(config_path, 
                                        num_labels=len(id2label), 
                                        max_position_embeddings=514,
                                        id2label=id2label,
                                        label2id=label2id)

    


   
    model = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True, ignore_mismatched_sizes=True, config=config)

    model.id2label = id2label
    model.label2id = label2id
    model.resize_token_embeddings(vocab_size)
    return model

def get_best(checkpoints_dir):
    checkpoints = os.listdir(checkpoints_dir)
    checkpoints = [i for i in checkpoints if "checkpoint" in i]
    checkpoints = [int(i.split("-")[1]) for i in checkpoints]
    checkpoints.sort()
    return os.path.join(checkpoints_dir,f"checkpoint-{checkpoints[0]}")


if __name__=='__main__':


    pass
    
    

