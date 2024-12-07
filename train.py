from module_utils import *
from pos_embed import *
from mathT import *
# from model import *
import torch.distributed as dist
from torch import *
import torch
import torch.nn as nn
from data import *
from llama import *
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = 'localhost'
    # os.environ["MASTER_PORT"] = "01234"

    torch.distributed.init_process_group('nccl')
    # dist.init_process_group(backend='nccl')
def cleanup():
    torch.distributed.destroy_process_group()
    
setup(0,1)
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
rank = torch.device("cuda", local_rank)
print('-----------------------------------success init ddp-----------------')

trainx,trainy1,trainy2,testx,testy1,testy2=data_preprocess("amass_shape/joint.pt","amass_shape/pose.pt","amass_shape/vacc.pt","amass_shape/vrot.pt")

dataset_train=Data_stage1(trainx,trainy1,trainy2,num_frame=10)
sampler = torch.utils.data.distributed.DistributedSampler(dataset_train,shuffle=False)
dataset_train = DataLoader(dataset=dataset_train, batch_size=32,sampler=sampler)

dataset_test=Data_stage1(testx,testy1,testy2,num_frame=10)
test_sampler= torch.utils.data.distributed.DistributedSampler(dataset_test,shuffle=False)
dataset_test = DataLoader(dataset=dataset_test, batch_size=16,sampler=test_sampler)

# model=LLAMA(48,10,128,10,39,4,10).to(rank)

model=LLAMA(48,10,128,5,39,2,4).to(rank)
loss = nn.MSELoss()
import logging
from utils import *
logging.basicConfig(
    filename="/root/autodl-tmp/128dim_10frame_4layer_lossweighted.log",
    format="%(asctime)s - %(name)s - %(levelname)s -%(module)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %p",
    level=logging.INFO,
)
model = DDP(model, device_ids=[rank],find_unused_parameters=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
early_stopping = EarlyStopping(patience=4, verbose=True)
for i in range (0,1000):
    model.train()
    logging.info("epoch {}----------------------------------".format(i))
    
    valid_loss_list=[]
    valid_loss_list2=[]
    valid_loss_all=[]
    c=0
    for x,y1,y2 in tqdm(dataset_train):
        x=x.flatten(2)
        
        x=x.permute(0,2,1)
        # print(x.shape)
        o1,o2,_=model(x.to(rank))
        # print(o1.shape)
        # print(y1.shape)
        loss1=loss(o1,y1.flatten(2).permute(0,2,1).to(rank))
        loss2=loss(o2,y2.flatten(2).permute(0,2,1).to(rank))
        loss_all=(loss1+loss2)*0.1
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        c=c+1
        if c%5000==0:
            logging.info("iter {}----------------------------------".format(c))
        # break
    with torch.no_grad():
        model.eval()
        for x,y1,y2 in tqdm(dataset_test):
            x=x.flatten(2)
            x=x.permute(0,2,1)
            o1,o2,_=model(x.to(rank))
            loss1=loss(o1,y1.flatten(2).permute(0,2,1).to(rank))
            loss2=loss(o2,y2.flatten(2).permute(0,2,1).to(rank))
            loss_all=(loss1+loss2)*0.1
            valid_loss_list.append(loss1.to('cpu'))
            valid_loss_list2.append(loss2.to('cpu'))
            valid_loss_all.append(loss_all.to('cpu'))
            # break
    logging.info("pose loss is {}".format(np.average(valid_loss_list)))
    logging.info("rotate loss is {}".format(np.average(valid_loss_list2)))
    
    avg_valid_loss = np.average(valid_loss_all)
    early_stopping(avg_valid_loss, model,path='./128dim_10frame_4layer_lossweighted.pth')
    if early_stopping.early_stop:
        
         break
    # print("test success")
    # break