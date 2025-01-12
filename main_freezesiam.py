import argparse
import time
import math
from os import path, makedirs

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torchvision import datasets
from torchvision import transforms

from model.loader import TwoCropsTransform
from model.model_factory import SimSingle
from model.criterion import SimSingleLoss
from model.validation import KNNValidation
from visualize import Visualize

parser = argparse.ArgumentParser('arguments for training')
parser.add_argument('--data_root', default="./Cifar10", type=str, help='path to dataset directory')
parser.add_argument('--exp_dir', default="./checkpoint", type=str, help='path to experiment directory')
parser.add_argument('--trial', type=str, default='freezesiam_v5', help='trial id')
parser.add_argument('--img_dim', default=32, type=int)

parser.add_argument('--arch', default='resnet18', help='model name is used for training')

parser.add_argument('--feat_dim', default=2048, type=int, help='feature dimension')
parser.add_argument('--num_proj_layers', type=int, default=2, help='number of projection layer')
parser.add_argument('--batch_size', type=int, default=192, help='batch_size')
parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--print_freq', default=100, type=int, help='print frequency')
parser.add_argument('--eval_freq', default=5, type=int, help='evaluate model frequency')
parser.add_argument('--save_freq', default=50, type=int, help='save model frequency')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
parser.add_argument("--freeze_model", default=None, type=str, help="path to freeze model")
parser.add_argument("--visualize", default=1, type=int, help="model visualize; no vis:0  freeze model vis:1  model vis:2")
parser.add_argument("--visualize_num", default=100, type=int, help="visualize data num for each label")

parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

args = parser.parse_args()

def dataset_statistics():
    train_set = datasets.CIFAR10(root=args.data_root,
                                 train=True,
                                 download=True)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=len(train_set),
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    for data, label in train_loader:
        print(data.shape)
        mean = torch.mean(data, dim=(0,2,3))
        std = torch.std(data, dim=(0,2,3))
        print(mean, std)


def main():
    if not path.exists(args.exp_dir):
        makedirs(args.exp_dir)

    trial_dir = path.join(args.exp_dir, args.trial)
    logger = SummaryWriter(trial_dir)
    print(vars(args))


    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(args.img_dim, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_set = datasets.CIFAR10(root=args.data_root,
                                 train=True,
                                 download=True,
                                 transform=TwoCropsTransform(train_transforms))

    train_loader = DataLoader(dataset=train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)

    model = SimSingle(args)
    freeze_model = SimSingle(args)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    criterion = SimSingleLoss()

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        freeze_model = freeze_model.cuda(args.gpu)
        criterion = criterion.cuda(args.gpu)
        cudnn.benchmark = True

    start_epoch = 1
    if args.resume is not None:
        if path.isfile(args.resume):
            start_epoch, model, optimizer = load_checkpoint(model, optimizer, args.resume)
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, start_epoch))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    if args.freeze_model is not None:
        if path.isfile(args.freeze_model):
            _, freeze_model, _ = load_checkpoint(freeze_model, optimizer, args.freeze_model)
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.freeze_model, start_epoch))
        else:
            print("No checkpoint found at '{}'".format(args.freeze_model))

    if args.visualize == 1:
        vis = Visualize(freeze_model.encoder, args)
        vis.visualize()
        return

    # routine
    best_acc = 0.0
    validation = KNNValidation(args, model.encoder)
    for epoch in range(start_epoch, args.epochs+1):

        adjust_learning_rate(optimizer, epoch, args)
        print("Training...")

        # train for one epoch
        train_loss = train(train_loader, model, freeze_model, criterion, optimizer, epoch, args)
        logger.add_scalar('Loss/train', train_loss, epoch)

        if epoch % args.eval_freq == 0:
            print("Validating...")
            val_top1_acc = validation.eval()
            print('Top1: {}'.format(val_top1_acc))

            # save the best model
            if val_top1_acc > best_acc:
                best_acc = val_top1_acc

                save_checkpoint(epoch, model, optimizer, best_acc,
                                path.join(trial_dir, '{}_best.pth'.format(args.trial)),
                                'Saving the best model!')
            logger.add_scalar('Acc/val_top1', val_top1_acc, epoch)

        # save the model
        if epoch % args.save_freq == 0:
            save_checkpoint(epoch, model, optimizer, val_top1_acc,
                            path.join(trial_dir, 'ckpt_epoch_{}_{}.pth'.format(epoch, args.trial)),
                            'Saving...')

    print('Best accuracy:', best_acc)

    if args.visualize == 2:
        vis = Visualize(model.encoder, args)
        vis.visualize()

    # save model
    save_checkpoint(epoch, model, optimizer, val_top1_acc,
                    path.join(trial_dir, '{}_last.pth'.format(args.trial)),
                    'Saving the model at the last epoch.')


def train(train_loader, model, freeze_model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        outs = model(images[0])
        freeze_outs = freeze_model(images[0])
        loss = criterion(outs['z1'], freeze_outs['z1'])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        losses.update(loss.item(), images[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

            # print(torch.nn.functional.cosine_similarity(outs['z1'], freeze_outs['z1'], dim=-1).mean())
            # # normalize repr. along the batch dimension
            # z_a_norm = (outs['z1'] - outs['z1'].mean(1, keepdim=True)) / outs['z1'].std(1, keepdim=True) # NxD

            N = outs['z1'].size(0)
            # D = outs['z1'].size(1)

            # # cross-correlation matrix
            # c = torch.abs(torch.mm(z_a_norm.T, z_a_norm)) / N # DxD
            # # multiply off-diagonal elems of c_diff by lambda
            # a = c[~torch.eye(D, dtype=bool)].mean()

            sim = torch.nn.functional.cosine_similarity(outs['z1'][None, :, :], freeze_outs['z1'][:, None, :], dim=-1)
            print(sim[~torch.eye(N, dtype=bool)].mean())

    return losses.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.learning_rate
    # cosine lr schedule
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(epoch, model, optimizer, acc, filename, msg):
    state = {
        'epoch': epoch,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'top1_acc': acc
    }
    torch.save(state, filename)
    print(msg)


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename, map_location='cuda:0')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return start_epoch, model, optimizer


if __name__ == '__main__':
    main()



