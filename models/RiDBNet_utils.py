import numpy as np
import torch
import torch.nn as nn
#from .group import QueryAndGroup
#from .subsample import furthest_point_sample
from torch_batch_svd import svd
from pointnet2_batch import pointnet2_cuda
from typing import Tuple
from torch.autograd import Function

# 用于在给定的点集 x 中找到每个点的 K 个最近邻
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


# 图特征   原本的
def get_graph_feature(x, k=20, idx=None):  # [4,3,1024]  
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k) [4,1024,20]
    device = torch.device('cuda')
    # 创建一个基础索引，用于将每个批次的索引映射到全局索引。这个操作将批次的索引扩展到每个点的位置
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    # 将基础索引加到最近邻的索引上，以获得全局索引
    idx = idx + idx_base # [4,1024,20]

    idx = idx.view(-1)# [4*1024*20]

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # [4,1024,3](batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]    # 根据全局索引提取 K 个最近邻的特征  [81920,3]
    feature = feature.view(batch_size, num_points, k, num_dims)  # [4,1024,20,3]
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) # [4,1024,20,3]
    # 将每个点的特征与其最近邻的特征进行合并。这里 feature - x 表示相对位置，而 x 表示原始位置
    feature = torch.cat((feature-x, x), dim=3).contiguous() # [4,6,1024,20]
    #  [4,1024,20,3]-[4,1024,20,3]   
    return feature


def grouping(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims).permute(0,3,1,2).contiguous() 
    return feature




# 获取激活函数类型
def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.2,inplace=True)
    else:
        return nn.ReLU(inplace=True)

class GroupingOperation(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample, device=features.device)

        pointnet2_cuda.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None


grouping_operation = GroupingOperation.apply

class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample, device=xyz.device).zero_()
        pointnet2_cuda.ball_query_wrapper(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 normalize_by_std=False,
                 normalize_by_allstd=False,
                 normalize_by_allstd2=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        """[summary]

        Args:
            radius (float): radius of ball
            nsample (int): maximum number of features to gather in the ball
            use_xyz (bool, optional): concate xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): [description]. Defaults to False.
            normalize_dp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.normalize_dp = normalize_dp
        self.normalize_by_std = normalize_by_std
        self.normalize_by_allstd = normalize_by_allstd
        self.normalize_by_allstd2 = normalize_by_allstd2
        assert self.normalize_dp + self.normalize_by_std + self.normalize_by_allstd < 2   # only nomalize by one method
        self.relative_xyz = relative_xyz
        self.return_only_idx = return_only_idx

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[
        torch.Tensor]:
        """
        :param query_xyz: (B, npoint, 3) xyz coordinates of the features
        :param support_xyz: (B, N, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        idx = ball_query(self.radius, self.nsample, support_xyz, query_xyz)

        if self.return_only_idx:
            return idx
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        if self.relative_xyz:
            grouped_xyz = grouped_xyz - query_xyz.transpose(1, 2).unsqueeze(-1)  # relative position
            if self.normalize_dp:
                grouped_xyz /= self.radius
        grouped_features = grouping_operation(features, idx) if features is not None else None
        return grouped_xyz, grouped_features


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set (idx)
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        # output = torch.cuda.IntTensor(B, npoint, device=xyz.device)
        # temp = torch.cuda.FloatTensor(B, N, device=xyz.device).fill_(1e10)
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2_cuda.furthest_point_sampling_wrapper(
            B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class PointsetGrouper_formal(nn.Module):
    def __init__(self, channel, reduce, kneighbors, radi, use_xyz=True, normalize="center", **kwargs):
        super(PointsetGrouper_formal, self).__init__()
        self.reduce = reduce
        self.kneighbors = kneighbors
        self.radi = radi
        self.use_xyz = use_xyz

        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            # add_channel=3 if self.use_xyz else 0
            add_channel=0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))
        self.ballquery = QueryAndGroup(self.radi, self.kneighbors)

    def forward(self, xyz, points): # [16,1024,3]  [16,1024,32] 
        points = points.transpose(2,1)
        B, N, C = points.shape  # [16,1024,64]
        S = xyz.shape[1]//self.reduce
        xyz = xyz.contiguous()
        fps_idx = furthest_point_sample(xyz, S).long() # [16,512]
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)

        grouped_xyz, grouped_points = self.ballquery(query_xyz=new_xyz, support_xyz=xyz, features=points.transpose(1,2).contiguous()) # [16,3,512,20] [16,64,512,20]
        grouped_xyz = grouped_xyz.permute(0, 2, 3, 1).contiguous()  # [16,512,20,3]
        grouped_points = grouped_points.permute(0, 2, 3, 1).contiguous()    # [16,512,20,64]

        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True) # [16,512,1,64]
            if self.normalize =="anchor":
                mean = new_points
                mean = mean.unsqueeze(dim=-2)
            grouped_points = (grouped_points-mean)   # [16,512,20,64]
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta  # [16,512,20,64]
        return new_xyz, grouped_points

    

class T_net(nn.Module):
    def __init__(self):
        super(T_net, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 1024)
        self.global_maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        B = x.shape[0]
        # --- 中心化处理 ---
        # mean = x.mean(dim=1, keepdim=True)
        # x_centered = x - mean
        x = self.relu(self.fc1(x.transpose(2, 1)))  # [B, 1024, 64]
        x = self.fc2(x)  # [B, 1024, 1024]
        x = self.global_maxpool(x.transpose(2, 1)).transpose(2, 1)
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x)).view(B, 3, 3)

        # --- L2 正则化：在输入矩阵上添加 εI ---
        epsilon = 1e-6
        I = torch.eye(3).to(x.device)
        x_reg = x + epsilon * I  

        q, r = torch.linalg.qr(x_reg)
        return q, r

# 最远点采样代替上面随机采样
def knn_sample_xyz_normal(xyz,normal,n_sample): 
    '''
    b,n,c  ->   b,n,c
    '''
    xyz = xyz.contiguous()
    normal = normal.contiguous()
    fps_idx = furthest_point_sample(xyz, n_sample).long() # [16,512]
    new_xyz = index_points(xyz, fps_idx)
    new_normal = index_points(normal, fps_idx)
    return new_xyz, new_normal
  
    
class RiDBNetSetAbstraction(nn.Module):
    def __init__(self, in_c=3,out_c=16,embed_channel=0,out_channel=64,gp=2,use_xyz=False, n_sample=512,k=24,radi=0.1,first=False,
                 normalize="anchor",reduce=2):
        super(RiDBNetSetAbstraction, self).__init__()
        self.k = k
        self.first=first
        self.grouper = PointsetGrouper_formal(out_c,reduce,self.k,radi,use_xyz,normalize)
        if not self.first:
            self.extract_feat = nn.Sequential(nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(out_c),
                                        nn.ReLU())
        self.n_sample = n_sample
        self.feature_fusion = feature_fusion_cls(out_c,embed_channel,out_channel,gp)
    def forward(self, x_global, group_points,x_local, normal_local,feature):
        ''' [16,1024,3] [16,6,1024] [16,1024,3] [16,1024,3] [16,64,1024]
        x_global:b,n,c 16,1024,3  16,1024,3
        group_points   16,3,1024  16,6,1024
        x_local:b,c,n  16,1024,3  16,1024,3
        normal_local   16,1024,3  16,1024,3
        feature:b,c,n             16,64,1024

        '''
        B, N, _= x_local.shape    # [16,1024,3]
        if self.first == True:
            group_points = get_graph_feature(group_points,self.k) # [16,1024,24,6]
            new_x_group = x_global
        else :
            group_points = self.extract_feat(group_points)  # [16,16,1024]
            new_x_group, group_points = self.grouper(x_global,group_points) # [16,1024,3] [16,1024,24,16]

        # [16,1024,3] [16,1024,3] [16,1024,24,22] [16,1024,24]
        new_x, new_normal,TIF_feature,idx_order = sample_and_group(x_local, normal_local, self.n_sample, self.k)
        TIF_feature = TIF_feature.contiguous() # [16,512,24,14]

        if feature is not None: 
            if idx_order is not None:
                grouped_feature = index_points_feature(feature.transpose(2,1), idx_order)  # [16,512,24,32]
            else:
                grouped_feature = feature.view(B, 1, N, -1)
            new_feature = torch.cat([TIF_feature, grouped_feature, group_points], dim=-1).permute(0,3,2,1)   # [16,64+22+16]
        else:
            new_feature = torch.cat([TIF_feature, group_points], dim=-1).permute(0,3,2,1) # [16,22+6,24,1024]
        group_points_max = torch.max(group_points, dim=2)[0].transpose(2,1)  # [16,6,1024]
        new_points = self.feature_fusion(new_feature)
        return new_x_group, group_points, group_points_max, new_x, new_normal, new_points # [16,1024,3] [16,6,1024] [16,1024,3] [16,1024,3] [16,64,1024]

class feature_fusion_cls(nn.Module):
    def __init__(self, out_c=16,embed_channel=0,out_channel=64,gp=2):
        super(feature_fusion_cls, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(22+embed_channel+out_c, out_channel, (1, 1), bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU())
        self.Pre = ResidualMLP(out_channel,mode='2d')
        self.Channel_attention = ECALayer(channels=out_channel)
        self.Spatial_attentione = G_Transformer(channels=out_channel,gp=gp)
        self.Pos=ResidualMLP(out_channel,mode='1d')
    def forward(self, x):
        new_feature = self.conv(x) # [16,64,24,512] 
        new_feature = self.Pre(new_feature) # [16,64,24,512] 
        Pre_feature = torch.max(new_feature, dim=2)[0] # [16,64,1024]
        channel_att_feature=self.Channel_attention(Pre_feature) # [16,64,1024] 
        spatial_att_feature=self.Spatial_attentione(Pre_feature)# [16,64,1024]
        new_feature=channel_att_feature+spatial_att_feature  # [16,64,1024]
        Pos_feature=self.Pos(new_feature) # [16,64,1024]
        new_points = Pos_feature + Pre_feature # [16,64,1024] 
        return new_points


class RiDBNetFeaturePropagation(nn.Module):
    def __init__(self,out_c=32,embed_channel=512,out_channel1=512,in_channel=256,out_channel2=512,gp=2,k=24):
        super(RiDBNetFeaturePropagation, self).__init__()
        self.k = k
        self.feature_fusion = feature_fusion_seg(out_c,embed_channel,out_channel1,in_channel,out_channel2,gp)
    def forward(self, new_x, new_normal,x,normal,feature1,feature2,group_points):
        ''' 

        '''
        TIF_feature,idx_order = sample_and_group_deconv(self.k, new_x, new_normal, x, normal)
        TIF_feature = TIF_feature.contiguous()
        group_feature = index_points_feature(feature1.transpose(2,1),idx_order)
        new_feature = torch.cat([TIF_feature,group_feature,group_points],dim=-1).permute(0,3,2,1)
        new_points = self.feature_fusion(new_feature,feature2)
        return new_points 




class feature_fusion_seg(nn.Module):
    def __init__(self, out_c=32,embed_channel=512,out_channel1=512,in_channel=256,out_channel2=512,gp=2):
        super(feature_fusion_seg, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(22+embed_channel+out_c, out_channel1, (1, 1), bias=False),
                                   nn.BatchNorm2d(out_channel1),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(out_channel1+in_channel, out_channel2, 1, bias=False),
                                   nn.BatchNorm1d(out_channel2),
                                   nn.ReLU())
        self.Pre = ResidualMLP(out_channel1,mode='2d')
        self.Channel_attention = ECALayer(channels=out_channel1)
        self.Spatial_attentione = G_Transformer(channels=out_channel1,gp=gp)
        self.Pos=ResidualMLP(out_channel1,mode='1d')
    def forward(self, x, feature):
        new_feature = self.conv1(x) # [16,64,24,512] 
        new_feature = self.Pre(new_feature) # [16,64,24,512] 
        Pre_feature = torch.max(new_feature, dim=2)[0] # [16,64,1024]
        channel_att_feature=self.Channel_attention(Pre_feature) # [16,64,1024] 
        spatial_att_feature=self.Spatial_attentione(Pre_feature)# [16,64,1024]
        new_feature=channel_att_feature+spatial_att_feature  # [16,64,1024]
        Pos_feature=self.Pos(new_feature) # [16,64,1024]
        new_points = Pos_feature + Pre_feature # [16,64,1024]
        result = torch.cat([new_points, feature],dim=1)
        result = self.conv2(result)
        return result


def sample_and_group_deconv(k, new_x, new_normal, x, normal):
    idx = knn_point(k, new_x, x)
    TIF_feature,idx_order = Ri_feature_LR_Risur(new_x,new_normal, x, normal,idx)
    return TIF_feature, idx_order

def torch_gather_nd(points,idx):
    '''
    Input:
        points: [B,N,C]
        idx:[B,nsample,K,2]  0:batch_idx 1:point_idx
    Return:
            [B,nsample,K,3]
    '''
    return points[idx[:,:,:,0],idx[:,:,:,1],:]

def index_points_feature(points, idx):
    """

    Input:
        points: input points data, [B, N, C] # [16,1024,64] [16,512,256]
        idx: sample index data, [B, S,K] # [16,512,24]  [16,256,24]
    Return:
        new_points:, indexed points data, [B, S, K,C] [16,512,24,64] [16,256,24,256]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long,device=points.device).view(view_shape).repeat(repeat_shape)
    idns_stack=torch.stack([batch_indices,idx.cuda()],dim=3)
    new_points = torch_gather_nd(points,idns_stack)
    return new_points

   
class ResidualMLP(nn.Module):
    def __init__(self,channel,mode='2d'):
        super(ResidualMLP, self).__init__()
        if mode=='2d':
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        elif mode=='1d':
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        else:
            raise NotImplementedError
        self.mode=mode
        self.net1=nn.Sequential(
            conv(in_channels=channel,out_channels=channel,kernel_size=1),
            bn(channel),
            nn.ReLU(),
        )
        self.net2=nn.Sequential(
            conv(in_channels=channel,out_channels=channel,kernel_size=1),
            bn(channel),
        )
        self.act=nn.ReLU()
    def forward(self,inputs):
        if self.mode=='2d':
            outputs=self.act(self.net2(self.net1(inputs)+inputs))
        else:
            outputs = self.act(self.net2(self.net1(inputs) + inputs))
        return outputs
    
# 注意力
class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv_avg = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        # feature descriptor on the global spatial information
        y_sparse_avg = torch.mean(x,dim=0)
        # Apply 1D convolution along the channel dimension
        y_avg = self.conv_avg(y_sparse_avg.unsqueeze(-1).transpose(-1, -2)
                      ).transpose(-1, -2).squeeze(-1)
        y=y_avg
        # y is (batch_size, channels) tensor

        # Multi-scale information fusion
        #y = self.sigmoid(y)
        y = self.relu(y)
        # y is (batch_size, channels) tensor
        # braodcast multiplication
        x=x*y
        return x

# 轻量化ECA，有待考察
class LightweightECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv_avg = nn.Conv1d(1, 1, kernel_size=k_size,
                                  padding=(k_size - 1) // 2, bias=False, groups=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 全局平均池化
        y_sparse_avg = torch.mean(x, dim=2, keepdim=True)
        # 1D 深度可分离卷积
        y_avg = self.conv_avg(y_sparse_avg.transpose(-1, -2)
                      ).transpose(-1, -2).squeeze(-1)
        # 激活函数
        y = self.sigmoid(y_avg)
        # 广播乘法
        x = x * y.unsqueeze(-1)
        return x

class G_Transformer(nn.Module):
    def __init__(self, channels, gp):
        super().__init__()
        mid_channels = channels
        self.gp = gp
        assert mid_channels % 4 == 0
        self.q_conv = nn.Conv1d(channels, mid_channels, 1, bias=False, groups=gp)
        self.k_conv = nn.Conv1d(channels, mid_channels, 1, bias=False, groups=gp)
        #self.q_conv.weight = torch.nn.Parameter(self.k_conv.weight.clone())
        #self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        # x=x.permute(0,2,1).contiguous()
        bs, ch, nums = x.size()
        x_q = self.q_conv(x)  # B x C x N
        x_q = x_q.reshape(bs, self.gp, ch // self.gp, nums)
        x_q = x_q.permute(0, 1, 3, 2)  # B x gp x num x C'

        x_k = self.k_conv(x)  # B x C x N
        x_k = x_k.reshape(bs, self.gp, ch // self.gp, nums)  # B x gp x C' x nums

        x_v = self.v_conv(x)
        energy = torch.matmul(x_q, x_k)  # B x gp x N x N
        energy = torch.sum(energy, dim=1, keepdims=False)

        attn = self.softmax(energy)
        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True))
        x_r = torch.matmul(x_v, attn)
        #x_r = self.act(self.after_norm(self.trans_conv(x_r)))
        x_r = self.act(self.after_norm(self.trans_conv(x-x_r))) # 模仿pct
        x = x + x_r
        return x.contiguous()

    

# class Point_Spatial_Attention(nn.Module): # attention模块
#     def __init__(self, in_dim):
#         super(Point_Spatial_Attention, self).__init__()

#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(16)
#         self.bn3 = nn.BatchNorm1d(16)
#         self.bn4 = nn.BatchNorm1d(in_dim)
        
#         self.mlp = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1, bias=False),
#                                 self.bn1,
#                                 nn.ReLU(),
                                
#                                 nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=False))
#         self.query_conv = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=16, kernel_size=1, bias=False),
#                                         self.bn2,
#                                         nn.ReLU())
                                        
#         self.key_conv = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=16, kernel_size=1, bias=False),
#                                         self.bn3,
#                                         nn.ReLU())
                                        
#         self.value_conv = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=in_dim, kernel_size=1, bias=False),
#                                         self.bn4,
#                                         nn.ReLU())


#         self.alpha = nn.Parameter(torch.zeros(1))
#         self.adaptive_weights = nn.Parameter(torch.ones(in_dim)) # 增加可学习权重
#         self.offset = nn.Parameter(torch.zeros(1))
#         self.softmax  = nn.Softmax(dim=-1)
        
#     def forward(self,x):
#         feat = self.mlp(x+self.offset) # [B, 128, 1024]  # 增加一个可偏移量
#         proj_query = self.query_conv(feat) # [B, 16, 1024]
#         proj_key = self.key_conv(feat).permute(0, 2, 1) # [B, 1024, 16]
#         similarity_mat = self.softmax(torch.bmm(proj_key, proj_query)) # [B, 1024, 1024]

#         proj_value = self.value_conv(feat) # [B, 3, 1024]

#         out = torch.bmm(proj_value * self.adaptive_weights.unsqueeze(0).unsqueeze(2), similarity_mat.permute(0, 2, 1))  # 增加可学习权重
#         #out = torch.bmm(proj_value, similarity_mat.permute(0, 2, 1))
#         out = self.alpha*out + x
#         return out


class Point_Spatial_Attention(nn.Module): # attention模块
    def __init__(self, in_dim):
        super(Point_Spatial_Attention, self).__init__()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(in_dim)
        
        self.mlp = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1, bias=False),
                                self.bn1,
                                nn.ReLU(),
                                nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=False))
        self.query_conv = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=16, kernel_size=1, bias=False),
                                        self.bn2,
                                        nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=16, kernel_size=1, bias=False),
                                        self.bn3,
                                        nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=in_dim, kernel_size=1, bias=False),
                                        self.bn4,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.offset = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        feat = self.mlp(x+self.offset) # [B, 128, 1024]
        proj_query = self.query_conv(feat).permute(0,2,1) # [B, 16, 1024]
        proj_key = self.key_conv(feat)# [B, 1024, 16]
        energy = proj_query @ proj_key
        similarity_mat = self.softmax(energy) # [B, 1024, 1024]
        
        similarity_mat = similarity_mat / (1e-9+similarity_mat.sum(dim=1, keepdims=True)) 

        proj_value = self.value_conv(feat) # [B, 3, 1024]
        #out = torch.bmm(proj_value, similarity_mat.permute(0, 2, 1)) # [b,3,1024]  原文
        out = proj_value@similarity_mat # [b,3,1024]
        out = self.alpha*out + x 
        return out


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist



def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    device = xyz.device
    B, N, C = xyz.size()
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    
    centroid = torch.mean(xyz, dim=1, keepdim=True) #[B, 1, C]
    dist = torch.sum((xyz - centroid) ** 2, -1)
    farthest = torch.max(dist, -1)[1]

    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def global_transform(points, npoints, train):
    points = points.permute(0, 2, 1).contiguous()
    idx = furthest_point_sample(points, npoints).long()
    centroids = index_points(points, idx)   #[B, S, C] 
    #centroids += 1e-6 * torch.randn_like(centroids)
    U, S, V = svd(centroids)
    if train == True:
        index = torch.randint(2, (points.size(0), 1, 3)).type(torch.FloatTensor).cuda()
        V_ = V * index
        V -= 2 * V_
    else:
        key_p = centroids[:, 0, :].unsqueeze(1)
        angle = torch.matmul(key_p, V)
        index = torch.le(angle, 0).type(torch.FloatTensor).cuda() 
        V_ = V * index
        V -= 2 * V_

    xyz = torch.matmul(points, V).permute(0, 2, 1) 

    return xyz



def compute_LRA(xyz, weighting=False, nsample = 64):
    dists = torch.cdist(xyz, xyz)

    dists, idx = torch.topk(dists, nsample, dim=-1, largest=False, sorted=False)
    dists = dists.unsqueeze(-1)

    group_xyz = index_points(xyz, idx)
    group_xyz = group_xyz - xyz.unsqueeze(2)

    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0
        M = torch.matmul(group_xyz.transpose(3,2), weights*group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3,2), group_xyz)

    # eigen_values, vec = M.symeig(eigenvectors=True)
    eigen_values, vec = torch.linalg.eigh(M, UPLO='U')

    LRA = vec[:,:,:,0]
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / LRA_length
    return LRA # B N 3

def compute_LRA_one(group_xyz, weighting=False):
    B, S, N, C = group_xyz.shape
    dists = torch.norm(group_xyz, dim=-1, keepdim=True) # nn lengths
    
    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0
        M = torch.matmul(group_xyz.transpose(3,2), weights*group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3,2), group_xyz)
    
    eigen_values, vec = torch.linalg.eigh(M, UPLO='U')
    
    LRA = vec[:,:,:,0]
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / LRA_length
    return LRA # B N 3

def order_index(xyz, new_xyz, new_norm, idx):
    # 根据每个点在平面上的投影方向与参考向量的夹角排序
    epsilon=1e-7
    B, S, C = new_xyz.shape
    nsample = idx.shape[2]
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, C)  # centered

    # project and order
    dist_plane = torch.matmul(grouped_xyz_local, new_norm)
    proj_xyz = grouped_xyz_local - dist_plane*new_norm.view(B, S, 1, C)
    proj_xyz_length = torch.norm(proj_xyz, dim=-1, keepdim=True)
    projected_xyz_unit = proj_xyz / (proj_xyz_length)
    projected_xyz_unit[projected_xyz_unit != projected_xyz_unit] = 0  # set nan to zero
    

    length_max_idx = torch.argmax(proj_xyz_length, dim=2)
    vec_ref = projected_xyz_unit.gather(2, length_max_idx.unsqueeze(-1).repeat(1,1,1,3)) # corresponds to the largest length
    
    dots = torch.matmul(projected_xyz_unit, vec_ref.view(B, S, C, 1))
    sign = torch.cross(projected_xyz_unit, vec_ref.view(B, S, 1, C).repeat(1, 1, nsample, 1))
    sign = torch.matmul(sign, new_norm)
    sign = torch.sign(sign)
    sign[:, :, 0, 0] = 1.  # the first is the vec_ref itself, should be 1.
    dots = sign*dots - (1-sign)
    dots_sorted, indices = torch.sort(dots, dim=2, descending=True)
    idx_ordered = idx.gather(2, indices.squeeze_(-1))

    return dots_sorted, idx_ordered



def calculate_surface_norm(surface_norm1,surface_norm2):
    norm_x=surface_norm1[:,:,:,1]*surface_norm2[:,:,:,2]-surface_norm1[:,:,:,2]*surface_norm2[:,:,:,1]
    norm_y=surface_norm1[:,:,:,2]*surface_norm2[:,:,:,0]-surface_norm1[:,:,:,0]*surface_norm2[:,:,:,2]
    norm_z=surface_norm1[:,:,:,0]*surface_norm2[:,:,:,1]-surface_norm1[:,:,:,1]*surface_norm2[:,:,:,0]
    norm=torch.cat([norm_x.unsqueeze(-1),norm_y.unsqueeze(-1),norm_z.unsqueeze(-1)],dim=-1)
    return norm

def calculate_two_surface_feature(x1,x1_norm,x2,x2_norm): # x2-->x1
    off_unit, surface_offset_length= calculate_unit(x1,x2) 
    cos_angle_1 = -(off_unit * x1_norm).sum(-1, keepdim=True) 
    cos_angle_2 = (off_unit * x2_norm).sum(-1, keepdim=True) 

    return off_unit,cos_angle_1, cos_angle_2,surface_offset_length

def calculate_unit(x1,x2):
    epsilon=1e-7
    offest_xi = x1 - x2
    surface_offest_length = torch.norm(offest_xi, dim=-1, keepdim=True) # L_offset
    offest_xi_unit = offest_xi / (surface_offest_length+epsilon)        # unit
    offest_xi_unit[offest_xi_unit != offest_xi_unit] = 0                # set nan to zero
    
    return offest_xi_unit,surface_offest_length


def Ri_feature_LR_Risur(xyz, norm, new_xyz, new_norm, idx, group_all=False):
    B, N, C = new_xyz.shape
    K = idx.shape[-1]
    dots_sorted, idx_ordered = order_index(xyz, new_xyz, new_norm.unsqueeze(-1), idx)
    
    grouped_xyz = index_points(xyz, idx_ordered)            # [B, npoint, nsample, C]
    xi_norm=index_points(norm, idx_ordered)                 # xi norm    
    if not group_all:
        xi = grouped_xyz - new_xyz.view(B, N, 1, C)
    else:
        xi = grouped_xyz 
    
    p_point = torch.zeros_like(xi)
    p_norm=(new_norm.unsqueeze(-2)).repeat([1,1,K,1])       # p norm

    num_shifts = 1
    if N>=1024:
        num_shifts = 2
    centroid_xyz = (torch.sum(grouped_xyz, dim=-2)) / N     
    center_pts = grouped_xyz - centroid_xyz.view(B,N,1,C)
    C_norm = compute_LRA_one(center_pts,weighting=True).view(B,N,1,C).repeat(1,1,K,1)     # centroid_norm
    centroid_xyz = centroid_xyz - new_xyz
    M = centroid_xyz.view(B, N, 1, C).repeat(1, 1, K, 1)    #
    
    x3=torch.roll(xi,shifts=num_shifts,dims=2)              # xi-1
    x3_norm=torch.roll(xi_norm,shifts=num_shifts,dims=2)    # xi-1_norm
    x4 = torch.roll(xi,shifts=-num_shifts,dims=2)           # xi+1
    x4_norm = torch.roll(xi_norm,shifts=-num_shifts,dims=2) # xi+1_norm


    x4_xi_unit, _   = calculate_unit(xi , x4)           # xi+1 --> xi
    x4_p_unit, _    = calculate_unit(p_point , x4)      # xi+1 --> p


    xi_p_unit , norm_anlge_1_1, norm_angle_2_1, length_0  = calculate_two_surface_feature(p_point,p_norm,xi,xi_norm)   #  alpha1 beta1   L1
    x3_p_unit , norm_angle_1_2, norm_angle_3_1, _         = calculate_two_surface_feature(p_point ,p_norm,x3,x3_norm)  #  alpha2 theta1
    x3_xi_unit, norm_angle_2_2, norm_angle_3_2, _         = calculate_two_surface_feature(xi ,xi_norm,x3,x3_norm)      #  beta2  theta2
    M_p_unit  , norm_angle_1_3, norm_angle_5_2, length_1  = calculate_two_surface_feature(p_point,p_norm,C,C_norm)     #  alpha3 w2      L2 
    _         , norm_angle_5_1, norm_angle_2_3, _         = calculate_two_surface_feature(M ,C_norm,xi, xi_norm)       #  w1     beta3
    norm_angle_4_1 = (x4_norm * x4_xi_unit).sum(-1, keepdim=True)  # gama1
    norm_angle_4_2 = (x4_norm * x4_p_unit).sum(-1, keepdim=True)   # gama2
    

    angle_1 = (xi_p_unit  * M_p_unit).sum(-1, keepdim=True)   # fai1
    angle_2 = (xi_p_unit  * x3_p_unit).sum(-1, keepdim=True)  # fai2
    angle_3 = (x4_p_unit  * xi_p_unit).sum(-1, keepdim=True)  # fai3
    angle_4 = (x3_p_unit  * x3_xi_unit).sum(-1, keepdim=True) # fai4
    angle_5 = (x4_p_unit  * x4_xi_unit).sum(-1, keepdim=True) # fai5

    
    surface_norm1 = calculate_surface_norm(x4_p_unit,xi_p_unit)       # pxi+1*pxi
    surface_norm2 = calculate_surface_norm(x3_p_unit,xi_p_unit)       # pxi-1*pxi
    surface_norm3 = calculate_surface_norm(M_p_unit,xi_p_unit)        # pc*pxi
    angle_6 = (surface_norm1 * surface_norm2).sum(-1, keepdim=True)   # fai6 = sin(fai2)sin(fai3)cos()  
    angle_7 = (surface_norm1 * surface_norm3).sum(-1, keepdim=True)   # fai7 = sin(fai3)sin(fai1)cos()
    angle_8 = (surface_norm2 * surface_norm3).sum(-1, keepdim=True)   # fai8 = sin(fai2)sin(fai1)cos()

    ri_feat = torch.cat([
            length_0,           #L1
            length_1,           #L2
            angle_1,            #fai1
            angle_2,            #fai2
            angle_3,            #fai3
            angle_4,            #fai4
            angle_5,            #fai5
            angle_6,            #fai6
            angle_7,            #fai7
            angle_8,            #fai8
            norm_anlge_1_1,     #alpha1
            norm_angle_1_2,     #alpha2
            norm_angle_1_3,     #alpha3
            norm_angle_2_1,     #beta1
            norm_angle_2_2,     #beta2
            norm_angle_2_3,     #beta3
            norm_angle_3_1,     #theta1
            norm_angle_3_2,     #theta2
            norm_angle_4_1,     #gama1
            norm_angle_4_2,     #gama2
            norm_angle_5_1,     #w1
            norm_angle_5_2,     #w2
            ], dim=-1)
    loc=torch.where(torch.isnan(ri_feat))
    ri_feat[loc]=1e-4
    return ri_feat, idx_ordered


def sample_and_group(x, norm, n_points, k): # [16,1024,3] [16,1024,3] 
    new_x, new_norm = knn_sample_xyz_normal(x, norm, n_points) #[16,512,3] [16,512,3]
    idx = knn_point(k, x, new_x) # [16,512,24]
    TIF_feature, idx_order = Ri_feature_LR_Risur(x, norm, new_x, new_norm, idx) # [16,1024,24,14] [16,1024,24]
    #TIF_feature = TIF_feature.permute(0,3,2,1).contiguous() # [16,14,24,1024]
    return new_x, new_norm, TIF_feature, idx_order  # [16,512,3] [16,512,3] [16,512,24,14] [16,512,24]
