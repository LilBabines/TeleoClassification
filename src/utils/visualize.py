from matplotlib import pyplot as plt
import json
import os

def plot_save_loss(result_path):
    '''Plot and save the loss and f1 score from the result_path/trainer_state.json file
    Args:
        result_path (str): The path to the result folder
    '''
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
            y_f1.append(item['eval_f1_macro_family'])



    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Train Val Loss, FISH Teleo 300 Medium 3 Mer")
    plt.plot(x_train_loss,y_train_loss,label='Train Loss')
    plt.plot(x_val_loss,y_val_loss,label='Val Loss')
    plt.grid(True)
    plt.legend()
    # plt.savefig(f'Images/LOSS_{result_path.split("/")[-1]}.png')
    # plt.savefig(f'{result_path}/LOSS.png')
    plt.show()


    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.title("Val Performance, FISH Teleo 300 Medium 3 Mer")
    #plt.plot(x_train_loss,y_train_loss,label='Train Loss')
    plt.plot(x_val_loss,y_f1)
    plt.grid(True)
    # plt.savefig(f'{result_path}/PERF.png')
    # plt.savefig(f'Images/PERF_{result_path.split("/")[-1]}.png')
    plt.show()