import json
from argparse import ArgumentParser

from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from preprocess import splits
from utils import load_data

parser = ArgumentParser()
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epochs', type=int, default=100)

args = parser.parse_args()

device = torch.device(args.device)


def run_epoch(epoch, model, data_loaders, loss_fn, optimizer):
    results = {}
    for split, data_loader in data_loaders.items():
        training = split == 'train'
        model.train(training)
        loss, acc, tot = [0] * 3
        with torch.set_grad_enabled(training):
            for input, label in tqdm(data_loader, ncols=80, desc=f'{split} epoch {epoch}'):
                input = input.to(device)
                label = label.to(device)
                if training:
                    optimizer.zero_grad()
                output = model.forward(input)
                pred = output.argmax(dim=1)
                batch_loss = loss_fn(output, label)
                loss += batch_loss.item()
                acc += (pred == label).sum().item()
                tot += len(label)
                if training:
                    batch_loss.backward()
                    optimizer.step()
        results[split] = {
            'loss': loss,
            'acc': acc / tot,
        }
    return results


if __name__ == '__main__':
    datasets = load_data()
    data_loaders = {split: DataLoader(datasets[split], args.batch_size, split == 'train') for split in splits}
    model = torch.hub.load('pytorch/vision:v0.7.0', 'resnet18', pretrained=True).to(device)
    model.fc = nn.Linear(model.fc.in_features, 4).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter()
    for epoch in range(1, args.epochs + 1):
        results = run_epoch(epoch, model, data_loaders, loss_fn, optimizer)
        print(json.dumps(results, indent=4, ensure_ascii=False))
        for split, result in results.items():
            for k, v in result.items():
                writer.add_scalar(f'{split}/{k}', v, epoch)
        writer.flush()
