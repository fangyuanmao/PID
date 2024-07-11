import argparse
import os
import copy
import json
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import HADARNet
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, HARDAloss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', default='./data/flir_train.h5',type=str)#, required=True)
    parser.add_argument('--eval-file', default='./data/flir_val.h5', type=str)#, required=True)
    parser.add_argument('--outputs-dir', default='./experiments/', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--num-epochs-save', type=int, default=50)
    parser.add_argument('--num-epochs-val', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--vnums', type=int, default=4)
    parser.add_argument('--model', type=str, default='HADARNet')
    parser.add_argument('--smp_model', type=str, default='pAN')
    parser.add_argument('--smp_encoder', type=str, default='resnet50')
    parser.add_argument('--smp_encoder_weights', type=str, default='imagenet')
    parser.add_argument('--losstype', type=str, default='hadarloss')
    args = parser.parse_args()

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    with open(os.path.join(args.outputs_dir,"params.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    if args.model == 'HADARNet':
        model = HADARNet(in_channels=3, out_channels=2+args.vnums, args=args).to(device)
    
    model = nn.DataParallel(model, device_ids=[0])
    if args.losstype == 'hadarloss':
        lossmodule = HARDAloss(vnums=args.vnums)
    optim_params = model.parameters()
    optimizer = optim.Adam(list(optim_params), lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=16)
    
    state_dict = model.state_dict()
    if args.resume is not None:
        for n, p in torch.load(args.resume, map_location=lambda storage, loc: storage)['state_dict'].items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
    
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = 1e8

    for epoch in range(args.start_epoch, args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            if epoch % args.num_epochs_save == 0 or epoch == args.num_epochs:
                torch.save({
                'state_dict': model.state_dict()}, os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))
            for data in train_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)
                loss = lossmodule.loss_rec(preds, labels)
                
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
                
            lossfile = open(os.path.join(args.outputs_dir,'loss.txt'),'a+')
            lossfile.write('epoch:'+str(epoch)+'; loss: {:.8f}'.format(epoch_losses.avg)+'\n')
            lossfile.close()
            torch.save({
                'state_dict': model.state_dict()}, os.path.join(args.outputs_dir, 'last.pth'))
        if epoch % args.num_epochs_val == 0 or epoch == args.num_epochs:
            model.eval()
            epoch_val_loss = AverageMeter()

            for data in eval_dataloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    preds = model(inputs)
                loss = lossmodule.loss_rec(preds, labels)
                epoch_val_loss.update(loss, len(inputs))

            print('eval val_loss: {:.8f}'.format(epoch_val_loss.avg))
            val_loss_file = open(os.path.join(args.outputs_dir,'val_loss.txt'),'a+')
            val_loss_file.write('epoch:'+str(epoch)+'; eval val_loss: {:.8f}'.format(epoch_val_loss.avg)+'\n')
            val_loss_file.close()

            if epoch_val_loss.avg < best_loss:
                best_epoch = epoch
                best_loss = epoch_val_loss.avg
                best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, val_loss: {:.4f}'.format(best_epoch, best_loss))
    val_loss_file = open(os.path.join(args.outputs_dir,'val_loss.txt'),'a+')
    val_loss_file.write('best epoch: {}, val_loss: {:.4f}'.format(best_epoch, best_loss)+'\n')
    val_loss_file.close()
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))