import argparse
import os
import copy
import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import TeVNet
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, TeVloss


def save_json(data, path):
    with open(path, mode="w") as f:
        json.dump(data, f, indent=4)


def save_model_checkpoint(state_dict, path):
    torch.save({'state_dict': state_dict}, path)


def save_loss_to_file(loss, epoch, path):
    with open(path, 'a+') as f:
        f.write(f'epoch: {epoch}; loss: {loss:.8f}\n')


def main(args):
    # Create output directory and save parameters
    os.makedirs(args.outputs_dir, exist_ok=True)
    save_json(vars(args), os.path.join(args.outputs_dir, "params.json"))

    # Set device and initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TeVNet(in_channels=3, out_channels=2 + args.vnums, args=args).to(device)
    model = nn.DataParallel(model, device_ids=[0])

    # Load model weights if resuming
    if args.resume:
        state_dict = model.state_dict()
        for n, p in torch.load(args.resume, map_location=lambda storage, loc: storage)['state_dict'].items():
            if n in state_dict:
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)

    # Set up loss function and optimizer
    lossmodule = TeVloss(vnums=args.vnums)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Data loaders
    train_dataset = TrainDataset(img_dir=args.train_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    eval_dataset = EvalDataset(img_dir=args.eval_dir)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=args.num_workers)

    # Training and evaluation loop
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = float('inf')

    for epoch in range(args.start_epoch, args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=len(train_loader)) as t:
            t.set_description(f'Epoch: {epoch}/{args.num_epochs - 1}')

            # Save model checkpoint
            if epoch % args.num_epochs_save == 0 or epoch == args.num_epochs - 1:
                save_model_checkpoint(model.state_dict(), os.path.join(args.outputs_dir, f'epoch_{epoch}.pth'))

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                preds = model(inputs)
                loss = lossmodule.loss_rec(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss=f'{epoch_losses.avg:.6f}')
                t.update(len(inputs))

            save_loss_to_file(epoch_losses.avg, epoch, os.path.join(args.outputs_dir, 'loss.txt'))
            save_model_checkpoint(model.state_dict(), os.path.join(args.outputs_dir, 'last.pth'))

        # Validation
        if epoch % args.num_epochs_val == 0 or epoch == args.num_epochs - 1:
            model.eval()
            epoch_val_loss = AverageMeter()

            with torch.no_grad():
                for inputs, labels in eval_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    preds = model(inputs)
                    loss = lossmodule.loss_rec(preds, labels)
                    epoch_val_loss.update(loss.item(), len(inputs))

            print(f'Validation Loss: {epoch_val_loss.avg:.8f}')
            save_loss_to_file(epoch_val_loss.avg, epoch, os.path.join(args.outputs_dir, 'val_loss.txt'))

            if epoch_val_loss.avg < best_loss:
                best_loss = epoch_val_loss.avg
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())

    # Final outputs
    print(f'Best epoch: {best_epoch}, Validation loss: {best_loss:.4f}')
    save_loss_to_file(best_loss, best_epoch, os.path.join(args.outputs_dir, 'val_loss.txt'))
    save_model_checkpoint(best_weights, os.path.join(args.outputs_dir, 'best.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='./data/train', type=str)
    parser.add_argument('--eval-dir', default='./data/test', type=str)
    parser.add_argument('--outputs-dir', default='./experiments/', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--num-epochs-save', type=int, default=50)
    parser.add_argument('--num-epochs-val', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--vnums', type=int, default=4)
    parser.add_argument('--smp_model', type=str, default='PAN')
    parser.add_argument('--smp_encoder', type=str, default='resnet50')
    parser.add_argument('--smp_encoder_weights', type=str, default='imagenet')
    args = parser.parse_args()

    main(args)
