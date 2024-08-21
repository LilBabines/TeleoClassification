import numpy as np
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids


    if isinstance(predictions, tuple):
        predictions = predictions[0]
    # Ensure predictions are 2D
    if predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=1)
    elif predictions.ndim > 2:
        raise ValueError(f"Unexpected predictions shape: {predictions.shape}")
    
    return f1_metric.compute(predictions=predictions, references=labels, average='macro')  

def training_argument(output_path, learning_rate=1e-5, per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=20, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True, push_to_hub=False):
    arg=TrainingArguments(
        output_dir=output_path,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        push_to_hub=push_to_hub
        )
    return arg

def define_trainer(model, tokenizer, train_dataset, val_dataset, training_args):

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    
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