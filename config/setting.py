'''
author: Changsheng Zheng
file: ModelNet40.py
Data: 2025/4/25
'''

import torch
import torch.nn.functional as F

def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss

def optimizer_set(args,classifier):
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), 
                                    lr=args.learning_rate*100, 
                                    momentum=0.9,
                                    weight_decay=args.decay_rate)
    if args.optimizer == 'Adamw':
        optimizer = torch.optim.AdamW(classifier.parameters(), 
                                    lr=args.learning_rate, 
                                    betas=(0.9, 0.999), 
                                    eps=1e-08, 
                                    weight_decay=args.decay_rate, 
                                    amsgrad=True)
    return optimizer

def scheduler_set(args,optimizer,epoch):
    if args.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    if args.scheduler == 'CosineRestart':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epoch, eta_min=1e-9)
    if args.scheduler == 'Consine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=args.learning_rate)
    return scheduler
