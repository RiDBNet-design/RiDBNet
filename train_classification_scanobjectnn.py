import os
import sys
import torch
import numpy as np
import datetime
import torch.nn as nn
import logging
from data_utils.ScanObjectNNLoader import ScanObjectNN
from config.setting import cal_loss,optimizer_set,scheduler_set

import provider
import importlib
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', type=bool, default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='RiDBNet_scanobj', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=15, type=int, help='training on ModelNet10/40')
    parser.add_argument('--epochs', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--data_type', type=str, default='hardest', help='data type')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default='', help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_uniform_sample', type=bool, default=True, help='use uniform sampiling')
    parser.add_argument('--scheduler', type=str, default='Consine', 
                        help='scheduler for training')
    parser.add_argument('--use_normals', action='store_true', default=True, 
                        help='use normals')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    return parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def test(model, loader,args, num_class=40):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points = points.transpose(2, 1)
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()  
        pred, feature = classifier(points)
        if len(pred.shape) == 3:
            pred = pred.mean(dim=1)
        pred_choice = pred.data.max(1)[1]
        for i in range(len(target)):
            class_acc[target[i],0]+=1
            if target[i]==pred_choice[i]:
                class_acc[pred_choice[i],1]+=1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))
    class_acc[:, 2] = class_acc[:, 1] / class_acc[:, 0]
    in_average=sum(class_acc[:, 1])/sum(class_acc[:, 0])
    cla_average=sum(class_acc[:, 2])/len(class_acc[:, 2])
    return in_average, cla_average

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
    exp_dir = exp_dir.joinpath('classification_scanobj')
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
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    if args.data_type == 'OBJ_NOBG':
        data_path = '../data/scanobjectnn/main_split_nobg/'
    elif args.data_type == 'hardest' or 'OBJ_BG': 
        data_path = '../data/scanobjectnn/main_split/'
    else:
        raise NotImplementedError()

    test_dataset = ScanObjectNN(root=data_path, args=args, split='test')
    train_dataset = ScanObjectNN(root=data_path, args=args, split='train') 
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/%s_utils.py' % args.model.split('_')[0], str(exp_dir))
    shutil.copy('./train_classification_scanobjectnn.py', str(exp_dir))
    shutil.copy('./config/setting.py', str(exp_dir))


    classifier = model.get_model(num_class,npoint_n=1,use_normals=args.use_normals)
    criterion = cal_loss
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        #classifier = nn.DataParallel(classifier, device_ids=[0,1])
        classifier = classifier.cuda()
        #criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
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

        
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points = points.data.numpy()
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()
            pred, feature = classifier(points)           
            if len(pred.shape) == 3:
                target_2 = target.unsqueeze(-1).repeat(1,pred.shape[1])
                target_2 = target_2.view(-1, 1)[:, 0]
                pred_2 = pred.contiguous().view(-1, num_class)  # N*K, num_class
                loss = criterion(pred_2, target_2.long())
                pred = pred.mean(dim=1) # N, num_class
            else:
                loss = criterion(pred, target.long())

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1
        scheduler.step()
        train_instance_acc = np.mean(mean_correct)
        
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        log_string('Train loss: %f' % loss)
        log_string('lr: %f' % optimizer.param_groups[0]['lr'])
        

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, args, num_class=num_class)
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
