from torch import *
import torch
import torch.nn as nn
from utils import *
from torch.utils.data import Dataset, DataLoader

def data_preprocess(joint_path,pose_path,vacc_path,vrot_path,train_test_split=0.95):
    joint=torch.concat(torch.load(joint_path),0)
    pose=torch.concat(torch.load(pose_path),0)
    vacc=torch.concat(torch.load(vacc_path),0)
    vrot=torch.concat(torch.load(vrot_path),0)
    joint_train,joint_test=joint[:int(joint.shape[0]*0.95)],joint[int(joint.shape[0]*0.95):]
    pose_train,pose_test=pose[:int(pose.shape[0]*0.95)],pose[int(pose.shape[0]*0.95):]
    vacc_train,vacc_test=vacc[:int(vacc.shape[0]*0.95)],vacc[int(vacc.shape[0]*0.95):]
    vrot_train,vrot_test=vrot[:int(vrot.shape[0]*0.95)],vrot[int(vrot.shape[0]*0.95):]
    
    assert joint_train.shape[0]==pose_train.shape[0]==vacc_train.shape[0]==vrot_train.shape[0]
    assert joint_test.shape[0]==pose_test.shape[0]==vacc_test.shape[0]==vrot_test.shape[0]
    # return [vacc_train,vrot_train],[joint_train,pose_train],[vacc_test,vrot_test],[joint_test,pose_test]
    return torch.cat((vacc_train,vrot_train.flatten(2)),2),joint_train,pose_train,torch.cat((vacc_test,vrot_test.flatten(2)),2),joint_test,pose_test


class Data_stage1(Dataset):
    def __init__(self,x,y1,y2,num_frame=20,index_set=index_set):
        self.x = x[:,index_set.estimate_imu_idx]
        self.y1=y1[:,index_set.estimate_smpl_idx]
        self.y2=y2[:,index_set.estimate_smpl_idx]
        self.num_frame=num_frame
    def __len__(self):
        return len(self.x)-self.num_frame
    
    def __getitem__(self, idx):
        returnx=self.x[idx:idx+self.num_frame]
        # return returnx.flatten(1).permute(0,2,1),self.y1[idx+self.num_frame],self.y2[idx+self.num_frame]
        return self.x[idx:idx+self.num_frame].permute(0,2,1),self.y1[idx:idx+self.num_frame],self.y2[idx:idx+self.num_frame]
