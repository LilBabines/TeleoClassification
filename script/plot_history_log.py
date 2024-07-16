from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import argparse
import os

exclude_keys = [
    "epoch",
    "step",
    "total_flos",
    "train_loss",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
]
result_path=r"C:\Users\Auguste Verdier\Desktop\ADNe\BouillaClip\Model\genera_300_medium_3_mer\checkpoint-85335"
with open(os.path.join(result_path,'trainer_state.json'), 'r') as f:
    data = json.load(f)
    log=data['log_history']
    #print(type(log))
x_train_loss=[]
y_train_loss=[]

x_val_loss=[]
y_val_loss=[]

y_f1=[]

for item in log:
    
    if 'loss' in item.keys():

        x_train_loss.append(item['epoch'])

        y_train_loss.append(item['loss']) 
        
    else :

        x_val_loss.append(item['epoch'])

        y_val_loss.append(item['eval_loss']) 
        y_f1.append(item['eval_f1'])



plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Train Val Loss, FISH Teleo 300 Medium 3 Mer")
plt.plot(x_train_loss,y_train_loss,label='Train Loss')
plt.plot(x_val_loss,y_val_loss,label='Val Loss')
plt.grid(True)
plt.legend()
plt.show()


plt.figure()
plt.xlabel('Epoch')
plt.ylabel('F1 score')
plt.title("Val Performance, FISH Teleo 300 Medium 3 Mer")
#plt.plot(x_train_loss,y_train_loss,label='Train Loss')
plt.plot(x_val_loss,y_f1)
plt.grid(True)
plt.legend()
plt.show()