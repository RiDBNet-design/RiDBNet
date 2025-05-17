import os
import sys
import torch
import numpy as np
import torch.nn as nn
import datetime
import logging
import provider
import importlib
import shutil
from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from config.setting import cal_loss,optimizer_set,scheduler_set
import argparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--use_cpu', action='store_true', default=False, 
                        help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0,1', 
                        help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='batch size in training')
    parser.add_argument('--model', default='RiDBNet_modelnet40', 
                        help='model name [default_lr_lgr: pointnet2_cls_msg]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  
                        help='training on ModelNet10/40')
    parser.add_argument('--epochs', default=300, type=int, 
                        help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, 
                        help='learning rate in training')
    parser.add_argument('--num_points', type=int, default=1024, 
                        help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', 
                        help='optimizer for training')
    parser.add_argument('--So3', type=bool, default=False,  
                        help='So3')


    parser.add_argument('--log_dir', type=str, default='', 
                        help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, 
                        help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=True, 
                        help='use normals')
    parser.add_argument('--process_data', action='store_true', default=True, 
                        help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, 
                        help='use uniform sampiling')
    parser.add_argument('--scheduler', type=str, default='Consine', 
                        help='scheduler for training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points,target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()
        
        points = points.transpose(2, 1)
        pred,feature = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc

import random
def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()     
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string(timestr)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '../data/modelnet40_normal_resampled/'

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data,SO3=args.So3)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data,SO3=args.So3)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,num_workers=8)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/%s_utils.py' % args.model.split('_')[0], str(exp_dir))
    shutil.copy('./train_classification_modelnet40.py', str(exp_dir))
    shutil.copy('./config/setting.py', str(exp_dir))

    classifier = model.get_model(num_class, npoint_n = 2,use_normals=args.use_normals,if_train=True)
    print(classifier)
    criterion = cal_loss
    classifier.apply(inplace_relu)
    if not args.use_cpu:
        #classifier = nn.DataParallel(classifier, device_ids=[0,1])
        classifier = classifier.cuda()
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    try:
        checkpoint = torch.load(str(exp_dir) + '/model/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
    
    optimizer = optimizer_set(args, classifier)
    scheduler = scheduler_set(args, optimizer, args.epochs)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    log_string('Trainable Parameters: %f' % (count_parameters(classifier)))
    
    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epochs):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epochs))
        mean_correct = []
        classifier = classifier.train()

        count = 0.0
        for batch_id, (points,target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            batch_size = points.size()[0]
            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points[:, :, 0:6] = provider.rotate_point_cloud_with_normal(points[:, :, 0:6])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points,target = points.cuda(), target.cuda()

            pred, feature = classifier(points)
            loss = criterion(pred, target.long())
            train_loss = loss.item()*batch_size
            count += batch_size
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
        scheduler.step()
        train_instance_acc = np.mean(mean_correct)

        log_string('Train Instance Accuracy: %f, loss: %.6f' % (train_instance_acc,train_loss*1.0/count))

        with torch.no_grad():
            test_classifier = model.get_model(num_class, npoint_n = 2,use_normals=args.use_normals,if_train=False).cuda()
            #test_classifier = nn.DataParallel(test_classifier, device_ids=[0,1])
            
            test_classifier.load_state_dict(classifier.state_dict())
            instance_acc, class_acc = test(test_classifier, testDataLoader, num_class=num_class)
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1


            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')
if __name__ == '__main__': 
    args = parse_args()
    main(args)
