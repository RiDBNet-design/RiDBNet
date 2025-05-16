import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RiDBNet_utils import *

class get_model(nn.Module):
    def __init__(self, num_category=40,npoint_n=2,use_normals=True,if_train=False):
        super(get_model, self).__init__()
        self.use_normals = use_normals
        self.attt = Point_Spatial_Attention(3)
        self.if_train = if_train
        self.sc0 = RiDBNetSetAbstraction(in_c=0,  out_c=6,  embed_channel=0,   out_channel=64,  gp=2, n_sample=512*npoint_n, k=8,  radi=None, first=True)
        self.sc1 = RiDBNetSetAbstraction(in_c=6,  out_c=16, embed_channel=64,  out_channel=128, gp=4, n_sample=256*npoint_n, k=16, radi=0.2)
        self.sc2 = RiDBNetSetAbstraction(in_c=16, out_c=16, embed_channel=128, out_channel=256, gp=4, n_sample=128*npoint_n, k=32, radi=0.4)
        self.sc3 = RiDBNetSetAbstraction(in_c=16, out_c=32, embed_channel=256, out_channel=512, gp=8, n_sample=64 *npoint_n, k=64, radi=0.8)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_category)
        )
    def forward(self, x):
        if self.use_normals:      # 
            normal = x[:, 3:, :]  # [16,3,1024]
            x = x[:, :3, :]       #[16,3,1024]
        else:
            # compute the LRA and use as normal
            normal = None
        batch_size = x.size(0)    # [B,3,1024]
        formal_x = x.transpose(2,1) # [16,1024,3]
        x_global = global_transform(x, 32, self.if_train)  # [16,3,1024]
        x_global = self.attt(x_global)   # [16,3,1024]

        x_global,_,group_points_max,x,normal,new_points = self.sc0(x_global.transpose(2,1),x_global,formal_x,normal.transpose(2,1),None) # [16,1024,3] [16,6,1024] [16,1024,3] [16,1024,3] [16,64,1024]
        x_global,_,group_points_max,x,normal,new_points = self.sc1(x_global,group_points_max,x,normal,new_points) # [16,512,3] [16,32,512] [16,3,512] [16,3,512] [16,64,512]
        x_global,_,group_points_max,x,normal,new_points = self.sc2(x_global,group_points_max,x,normal,new_points) # [16,256,3] [16,64,256] [16,3,256] [16,3,256] [16,128,256]
        x_global,_,group_points_max,x,normal,new_points = self.sc3(x_global,group_points_max,x,normal,new_points) # [16,128,3] [16,128,128] [16,3,128] [16,3,128] [16,256,128]
        result = F.adaptive_max_pool1d(new_points, 1).view(batch_size, -1) #[16,256]
        result = self.classifier(result)
        return result, new_points