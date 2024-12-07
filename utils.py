import torch
import numpy as np

class EarlyStopping():
    def __init__(self,patience=5,verbose=False,delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self,val_loss,model,path):
        print("val_loss={}".format(val_loss))
        score = -val_loss #update loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
        elif score < self.best_score+self.delta: #if overfitting-> tirgger earlystopping
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
            self.counter = 0
    def save_checkpoint(self,val_loss,model,path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')#if not overfitting->save model 
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
        
        
class index_set:
    estimate_imu_idx=[0,1,4,5]
    num_estimate_imu=len(estimate_imu_idx)
    estimate_smpl_idx=[0,3,6,9,12,13,14,16,17,18,19,20,21]
    num_estimate_smpl=len(estimate_smpl_idx)
    
    generate_smpl_idx=[1,2,4,5,7,8]
    num_genrate_smpl=len(generate_smpl_idx)
