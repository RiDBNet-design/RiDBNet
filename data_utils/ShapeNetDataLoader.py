import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

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

def rotate_point_cloud(points, normals):
    rotation_angle = np.random.uniform(0, 1) * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    
    rotated_data = np.dot(points.reshape((-1, 3)), rotation_matrix)
    rotated_normal = np.dot(normals.reshape((-1, 3)), rotation_matrix)
    return rotated_data, rotated_normal


class PartNormalDataset(Dataset):
    def __init__(self,root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', 
                 class_choice=None, use_normals=False,z=False,SO3=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = use_normals
        self.z = z
        self.SO3=SO3

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        #np.random.seed(2)
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        if self.z:
            point_set[:, 0:3], point_set[:, 3:6] = rotate_point_cloud(point_set[:, 0:3], point_set[:, 3:6])

        if self.SO3:
            point_set = rotate_point_cloud_with_normal_so3_single(point_set)
            
        choice = np.random.choice(len(seg), self.npoints, replace=True) 
        point_set = point_set[choice, :]
        seg = seg[choice]
        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)



