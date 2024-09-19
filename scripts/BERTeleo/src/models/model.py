from torch import nn
import torch

import os
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModel, AutoModel, BertConfig

from transformers import AutoModelForSequenceClassification
from datasets import load_dataset




class MultiTaxaClassification(nn.Module):
    def __init__(self, num_labels_order = 72, num_labels_family = 303, hidden_size =768, vocab_size = None,**bert_kwargs ):
        super(MultiTaxaClassification,self).__init__()
        
        self.num_labels = (num_labels_order, num_labels_family)
        
        self.problem_type = "multi_label_classification"
        self.hidden_size = hidden_size = 768
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M",vocab_size= vocab_size,**bert_kwargs )
        self.model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True,config=config, ignore_mismatched_sizes=True)
        hidden_size = self.model.config.hidden_size
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
        
        

        outputs = self.model(
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

if __name__=='__main__':


    pass
    
    

