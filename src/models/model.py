from torch import nn

import os
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModel, AutoModel, BertConfig

from transformers import AutoModelForSequenceClassification
from datasets import load_dataset




class BertForSequenceMultiTaxaClassification(nn.Module):
    def __init__(self, model_name_or_path, num_labels_order = 72, num_labels_family = 303  ):
        super(BertForSequenceMultiTaxaClassification, self).__init__()
        
        self.num_labels = (num_labels_order, num_labels_family)
        
        self.problem_type = "multi_label_classification"

        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M",
                                    max_position_embeddings=514
        )
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True,config=config)
        hidden_size = self.model.config.hidden_size
        classifier_dropout = 0.1

        self.dropout = nn.Dropout(classifier_dropout)
        

        self.classifier_order = nn.Linear(hidden_size, self.num_labels[0])
        self.classifier_family = nn.Linear(hidden_size, self.num_labels[1])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels_order=None,
        labels_family=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        
        logits_order = self.classifier_order(pooled_output)
        logits_family = self.classifier_family(pooled_output)
        logits = (logits_order, logits_family)

        loss = None
        if labels_order is not None and labels_family is not None:
            
            loss_fct = nn.CrossEntropyLoss()
            loss_order = loss_fct(logits_order, labels_order)
            loss_family = loss_fct(logits_family, labels_family)
            loss = loss_order + loss_family
            outputs = (loss,) + outputs

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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




def load_model(name, vocab_size, local=False, id2label=None, label2id=None,dropout=False):

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

    if dropout:

        
        model = DNABERTWithDropout.from_pretrained(
            "zhihan1996/DNABERT-2-117M", 
            trust_remote_code=True, 
            ignore_mismatched_sizes=True, 
            config=config
        )

    else:
        model = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True, ignore_mismatched_sizes=True, config=config)

    model.id2label = id2label
    model.label2id = label2id
    model.resize_token_embeddings(vocab_size)
    return model

if __name__=='__main__':


    pass
    
    

