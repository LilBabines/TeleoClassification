import numpy as np
from transformers import Trainer, DataCollatorWithPadding
import evaluate
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
import torch

available_metrics = {
        'MulticlassF1Score': MulticlassF1Score,
        'MulticlassAccuracy': MulticlassAccuracy
    }
    

def define_metrics(num_classes,dict):
    metrics_family = {}
    metrics_order = {}

    if isinstance(num_classes, int):


        for metric_name, metric in dict.items():
        
            metric_class = available_metrics[metric.callable](num_classes=num_classes, **metric.kwargs)
        
            metrics_family[metric_name+'_family'] = metric_class
    else:
        for metric_name, metric in dict.items():
            
            metric_class = available_metrics[metric.callable](num_classes=num_classes[0], **metric.kwargs)
            metrics_order[metric_name+'_order'] = metric_class
            
            metric_class = available_metrics[metric.callable](num_classes=num_classes[1], **metric.kwargs)
            metrics_family[metric_name+'_family'] = metric_class
    return metrics_order,metrics_family




def define_trainer(model, tokenizer, train_dataset, val_dataset,num_classes,metrics, training_args):

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    metrics_dict_order, metrics_dict_family = define_metrics(num_classes,metrics)

    
    def to_tensor(data):
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, tuple):
            return tuple(to_tensor(d) for d in data)
        raise TypeError("Unsupported data type: {}".format(type(data)))
        
    def compute_metrics(eval_pred):

        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        output = {}

        predictions = to_tensor(predictions)
        labels = to_tensor(labels)

        if labels.ndim == 2:
            print("no")
            labels_order, labels_family= labels[:, 0], labels[:, 1]
            predictions_order , predictions_family = predictions
            for key, func in metrics_dict_order.items():
                output[key] = func(preds =predictions_order, target =labels_order)
            for key, func in metrics_dict_family.items():
                output[key] = func(preds =predictions_family, target =labels_family)

        else :
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            
            for key, func in metrics_dict_family.items():
                output[key] = func(preds =predictions, target =labels)
        
        
        
        return output

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )



    return trainer