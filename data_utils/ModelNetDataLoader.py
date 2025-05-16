import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
from models.RiDBNet_utils import compute_LRA,furthest_point_sample

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def rotate_point_cloud_with_normal_so3_single(point_cloud):
    """ Randomly perturb the point cloud by rotation
        Input:
          Nx6 array, original point cloud and point normals
        Return:
          Nx6 array, rotated point cloud and normals
    """
    rotated_data = np.zeros(point_cloud.shape, dtype=np.float32)
    
    # Generate random rotation angles
    rotation_angle_A = np.random.uniform() * 2 * np.pi
    rotation_angle_B = np.random.uniform() * 2 * np.pi
    rotation_angle_C = np.random.uniform() * 2 * np.pi
    
    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotation_angle_A), -np.sin(rotation_angle_A)],
                   [0, np.sin(rotation_angle_A), np.cos(rotation_angle_A)]])
    
    Ry = np.array([[np.cos(rotation_angle_B), 0, np.sin(rotation_angle_B)],
                   [0, 1, 0],
                   [-np.sin(rotation_angle_B), 0, np.cos(rotation_angle_B)]])
    
    Rz = np.array([[np.cos(rotation_angle_C), -np.sin(rotation_angle_C), 0],
                   [np.sin(rotation_angle_C), np.cos(rotation_angle_C), 0],
                   [0, 0, 1]])
    
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    # Separate point cloud and normals
    shape_pc = point_cloud[:, 0:3]
    shape_normal = point_cloud[:, 3:6]
    
    # Apply rotation
    rotated_data[:, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
    rotated_data[:, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    
    return rotated_data


def rotate_point_cloud_so3(points):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx6 array, original batch of point clouds
        Return:
          BxNx6 array, rotated batch of point clouds
    """
    rotation_angle_A = np.random.uniform() * 2 * np.pi
    rotation_angle_B = np.random.uniform() * 2 * np.pi
    rotation_angle_C = np.random.uniform() * 2 * np.pi

    cosval_A = np.cos(rotation_angle_A)
    sinval_A = np.sin(rotation_angle_A)
    cosval_B = np.cos(rotation_angle_B)
    sinval_B = np.sin(rotation_angle_B)
    cosval_C = np.cos(rotation_angle_C)
    sinval_C = np.sin(rotation_angle_C)
    rotation_matrix = np.array([[cosval_B*cosval_C, -cosval_B*sinval_C, sinval_B],
                                [sinval_A*sinval_B*cosval_C+cosval_A*sinval_C, -sinval_A*sinval_B*sinval_C+cosval_A*cosval_C, -sinval_A*cosval_B],
                                [-cosval_A*sinval_B*cosval_C+sinval_A*sinval_C, cosval_A*sinval_B*sinval_C+sinval_A*cosval_C, cosval_A*cosval_B]])
    rotated_data = np.dot(points[:,:,0:3], rotation_matrix)
    rotated_normal = np.dot(points[:,:,3:6], rotation_matrix)
    return rotated_data, rotated_normal


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False,SO3=False):
        self.root = root
        self.npoints = args.num_points
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.SO3 = SO3

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = furthest_point_sample(point_set, self.npoints)
                        #point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                if  self.use_normals:
                    print('Load processed data from %s...' % self.save_path)
                    with open(self.save_path, 'rb') as f:
                        self.list_of_points, self.list_of_labels = pickle.load(f)
                else:
                    new_save_path = os.path.join(root, 'modelnet40_%s_1024pts_fps_without_normal.dat' % (split))
                    if  os.path.exists(new_save_path):
                        print("exist without normal dataset")
                        with open(new_save_path, 'rb') as f:
                            self.list_of_points, self.list_of_labels = pickle.load(f)
                    else:
                        with open(self.save_path, 'rb') as f:
                            self.list_of_points, self.list_of_labels = pickle.load(f)
                        self.list_of_points=torch.tensor(np.array(self.list_of_points)[:,:,:3]).cuda()
                        print("without normals, using LRA to compute normals")
                        # compute the LRA and use as normal
                        norm = torch.zeros_like(self.list_of_points).cuda()
                        
                        B=16
                        L=int(len(self.list_of_points)/B)                 
                        for index in tqdm(range(L), total=L):
                            norm[index*B:(index+1)*B,:,:] = compute_LRA(self.list_of_points[index*B:(index+1)*B,:,:], True, nsample = 32)
                        norm[(index+1)*B:,:,:] = compute_LRA(self.list_of_points[(index+1)*B:,:,:], True, nsample = 32)
                        self.list_of_points = torch.cat([self.list_of_points, norm], dim=-1)
                        self.list_of_points=self.list_of_points.cpu().numpy()
                        
                        with open(new_save_path, 'wb') as f:
                            pickle.dump([self.list_of_points.numpy(), self.list_of_labels], f)           


    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if self.SO3:
            point_set = rotate_point_cloud_with_normal_so3_single(point_set)
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('../data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
