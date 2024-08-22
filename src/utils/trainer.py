import numpy as np
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
import torch


    

        


def training_argument(**args): #output_path, learning_rate=1e-5, per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=20, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True, push_to_hub=False
    # arg=TrainingArguments(
    #     output_dir=output_path,
    #     learning_rate=learning_rate,
    #     per_device_train_batch_size=per_device_train_batch_size,
    #     per_device_eval_batch_size=per_device_eval_batch_size,
    #     num_train_epochs=num_train_epochs,
    #     weight_decay=weight_decay,
    #     eval_strategy=eval_strategy,
    #     save_strategy=save_strategy,
    #     load_best_model_at_end=load_best_model_at_end,
    #     metric_for_best_model=metric_for_best_model,
    #     greater_is_better=greater_is_better,
    #     push_to_hub=push_to_hub,
    #     use_cpu=True
    #     )
    arg = TrainingArguments(**args)
    return arg

def define_trainer(model, tokenizer, train_dataset, val_dataset,num_classes, training_args):

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if isinstance(num_classes, int):
        metrics_dict_family = {
        'f1_micro': MulticlassF1Score(num_classes=num_classes, average='micro'),
        'f1_macro': MulticlassF1Score(num_classes=num_classes, average='macro'),
        'accuracy': MulticlassAccuracy(num_classes=num_classes)
    }

    else :
        metrics_dict_order = {
            'f1_micro_order': MulticlassF1Score(num_classes=num_classes[0], average='micro'),
            'f1_macro_order': MulticlassF1Score(num_classes=num_classes[0], average='macro'),
            'accuracy_order': MulticlassAccuracy(num_classes=num_classes[0])
        }
        metrics_dict_family = {
            'f1_micro_family': MulticlassF1Score(num_classes=num_classes[1], average='micro'),
            'f1_macro_family': MulticlassF1Score(num_classes=num_classes[1], average='macro'),
            'accuracy_family': MulticlassAccuracy(num_classes=num_classes[1])
        }
    
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
            labels_order, labels_family= labels[:, 0], labels[:, 1]
            predictions_order , predictions_family = predictions
            for key, func in metrics_dict_order.items():
                output[key] = func(preds =predictions_order, target =labels_order)
            for key, func in metrics_dict_family.items():
                output[key] = func(preds =predictions_family, target =labels_family)

        else :
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