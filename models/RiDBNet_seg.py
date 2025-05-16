import torch
import torch.nn as nn
import torch.nn.functional as F
from models.RiDBNet_utils import *

class get_model(nn.Module):
    def __init__(self, num_category=50,npoint_n=1,use_normals=True,if_train=False):
        super(get_model, self).__init__()
        self.use_normals = use_normals
        self.attt = Point_Spatial_Attention(3)
        self.if_train = if_train
        self.npoint_n = npoint_n
        self.category_num=16
        self.sc0 = RiDBNetSetAbstraction(in_c=0,  out_c=6,  embed_channel=0,   out_channel=64,  gp=2, n_sample=512*npoint_n, k=20,  radi=None, first=True)
        self.sc1 = RiDBNetSetAbstraction(in_c=6,  out_c=16, embed_channel=64,  out_channel=128, gp=4, n_sample=256*npoint_n, k=20, radi=0.2)
        self.sc2 = RiDBNetSetAbstraction(in_c=16, out_c=32, embed_channel=128, out_channel=256, gp=4, n_sample=128*npoint_n, k=20, radi=0.4)
        self.sc3 = RiDBNetSetAbstraction(in_c=32, out_c=64, embed_channel=256, out_channel=512, gp=8, n_sample=64*npoint_n, k=20, radi=0.8)
        
        self.sc4 = RiDBNetFeaturePropagation(out_c=32, embed_channel=512, out_channel1=512, in_channel=256,  out_channel2=512,  gp=8, k=20)
        self.sc5 = RiDBNetFeaturePropagation(out_c=16, embed_channel=512, out_channel1=512, in_channel=128,  out_channel2=256,  gp=8, k=20)
        self.sc6 = RiDBNetFeaturePropagation(out_c=6,  embed_channel=256, out_channel1=256, in_channel=64,   out_channel2=128,  gp=4, k=20)
        self.sc7 = RiDBNetFeaturePropagation(out_c=6,  embed_channel=128, out_channel1=128, in_channel=16, out_channel2=128, gp=4, k=20)
        self.seg = nn.Sequential(
            nn.Dropout(0.4),
            nn.Conv1d(128, num_category, 1)
        )
    def forward(self, x, cls_label):
        if self.use_normals:      # 
            normal = x[:, 3:, :]  # [16,3,1024]
            x = x[:, :3, :]       #[16,3,1024]
        else:
            # compute the LRA and use as normal
            normal = None
        batch_size, _, N = x.size()   # [B,3,1024]
        formal_x = x.transpose(2,1) # [16,1024,3]
        x_global = global_transform(x, 32, self.if_train)  # [16,3,1024]
        x_global = self.attt(x_global).transpose(2,1).contiguous()   # [16,3,1024]
        group_points0 = get_graph_feature(x_global.transpose(2,1),20)
        fpsx_idx = furthest_point_sample(x_global,512).long()
        x_global = index_points(x_global,fpsx_idx).transpose(2,1)



        x_global1,group_points1,group_points_max1,x1,normal1,new_points1 = self.sc0(x_global.transpose(2,1),x_global,formal_x,normal.transpose(2,1),None) # [16,1024,3] [16,6,1024] [16,1024,3] [16,1024,3] [16,64,1024]
        x_global2,group_points2,group_points_max2,x2,normal2,new_points2 = self.sc1(x_global1,group_points_max1,x1,normal1,new_points1) # [16,512,3] [16,32,512] [16,3,512] [16,3,512] [16,64,512]
        x_global3,group_points3,group_points_max3,x3,normal3,new_points3 = self.sc2(x_global2,group_points_max2,x2,normal2,new_points2) # [16,256,3] [16,64,256] [16,3,256] [16,3,256] [16,128,256]
        x_global4,group_points4,group_points_max4,x4,normal4,new_points4 = self.sc3(x_global3,group_points_max3,x3,normal3,new_points3) # [16,128,3] [16,128,128] [16,3,128] [16,3,128] [16,256,128]
 
        new_points5 = self.sc4(x4,normal4,x3,normal3,new_points4,new_points3,group_points3)
        new_points6 = self.sc5(x3,normal3,x2,normal2,new_points5,new_points2,group_points2)
        new_points7 = self.sc6(x2,normal2,x1,normal1,new_points6,new_points1,group_points1)
        cls_label_one_hot = cls_label.view(batch_size,self.category_num,1).repeat(1,1,N).cuda()
        new_points8 = self.sc7(x1,normal1,formal_x,normal.transpose(2,1),new_points7,cls_label_one_hot,group_points0)
        
        result = self.seg(new_points8).transpose(2,1)
        return result, new_points8