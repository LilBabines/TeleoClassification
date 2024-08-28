from matplotlib import pyplot as plt
import json
import os
import glob


def plot_save_loss(path, metrics):
    '''Plot and save the loss and f1 score from the result_path/trainer_state.json file
    Args:
        result_path (str): The path to the result folder
    '''
    paths_checkpoints = glob.glob(os.path.join(path,'checkpoint-*'))
    sorted_paths_checkpoints = sorted(paths_checkpoints, key=lambda x: int(x.split('-')[-1]))
    result_path = sorted_paths_checkpoints[-1]
    
    with open(os.path.join(result_path,'trainer_state.json'), 'r') as f:
        data = json.load(f)
        log=data['log_history']
        #print(type(log))
    x_train_loss=[]
    y_train_loss=[]

    x_val_loss=[]
    y_val_loss=[]

    y_metrics={'eval_'+m : [] for m in metrics}

    for item in log:
        
        if 'loss' in item.keys():

            x_train_loss.append(item['epoch'])

            y_train_loss.append(item['loss']) 
            
        else :

            x_val_loss.append(item['epoch'])

            y_val_loss.append(item['eval_loss']) 
            for m in metrics:
                if 'eval_'+m in item.keys():
                    y_metrics['eval_'+m].append(item['eval_'+m])
                    



    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Train Val Loss")
    plt.plot(x_train_loss,y_train_loss,label='Train Loss')
    plt.plot(x_val_loss,y_val_loss,label='Val Loss')
    plt.grid(True)
    plt.legend()
    # plt.savefig(f'Images/LOSS_{result_path.split("/")[-1]}.png')
    # plt.savefig(f'{result_path}/LOSS.png')
    plt.savefig(f'{path}/LOSS.png')
    plt.show()


    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title("Val Performance")
    #plt.plot(x_train_loss,y_train_loss,label='Train Loss')
    for key in y_metrics.keys():
        plt.plot(x_val_loss,y_metrics[key],label=key)
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{path}/PERF.png')
    
    plt.show()