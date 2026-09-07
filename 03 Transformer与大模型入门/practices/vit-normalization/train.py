"""Train the supplied ViT/MixLN implementations on an ImageFolder dataset."""
from pathlib import Path
import argparse
import json
import random
import torch
from torch import nn
from utils.vit import ViT
from utils.mixln_vit import MixLNViT
from utils.scheduler import WarmupCosineSchedule

ROOT = Path(__file__).resolve().parents[3]


def make_model(variant='baseline', alpha=0.25, **overrides):
    config = dict(image_size=224, patch_size=32, num_classes=100, dim=392,
                  depth=12, heads=8, dim_head=49, mlp_dim=1568, dropout=0.1)
    config.update(overrides)
    if variant == 'baseline':
        return ViT(**config)
    if variant not in ('PostPre', 'PreB2TPost', 'PrePost'):
        raise ValueError('Unknown normalization variant')
    return MixLNViT(**config, alpha=alpha, Mix_type=variant)


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    model.train(optimizer is not None)
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            count += y.numel()
            loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(-1) == y).sum().item()
    if not count:
        raise ValueError('Empty dataset')
    return {'loss': loss_sum / count, 'accuracy': correct / count}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='ImageFolder: one subdirectory per class')
    p.add_argument('--variant', choices=['baseline','PostPre','PreB2TPost','PrePost'], default='baseline')
    p.add_argument('--alpha', type=float, default=0.25)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.03)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, default=ROOT/'outputs/vit-normalization/run')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0:
        p.error('epochs, batch-size and lr must be positive; workers must be nonnegative')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Output directory is nonempty; choose a new --output to preserve previous runs')
    if not args.data.is_dir():
        p.error('Missing ImageFolder dataset; see README')
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Subset
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # ImageFolder sorts class names, converts RGB, and handles native paths.
    train_data = ImageFolder(args.data, transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.RandAugment(), transforms.ToTensor()]))
    val_data = ImageFolder(args.data, transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))
    train_ids, val_ids = train_test_split(list(range(len(train_data))), test_size=0.1,
                                         stratify=train_data.targets, random_state=args.seed)
    loaders = [DataLoader(Subset(data, ids), batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.workers)
               for data, ids, shuffle in [(train_data,train_ids,True),(val_data,val_ids,False)]]
    model = make_model(args.variant,args.alpha,num_classes=len(train_data.classes)).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.9)
    steps = args.epochs * len(loaders[0])
    scheduler = WarmupCosineSchedule(optimizer,warmup_steps=min(1000,steps//10),t_total=steps)
    args.output.mkdir(parents=True)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config['class_to_idx'] = train_data.class_to_idx
    config['validation_files'] = [str(Path(val_data.samples[i][0]).relative_to(args.data)) for i in val_ids]
    (args.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    history=[]
    for epoch in range(args.epochs):
        train = run_epoch(model,loaders[0],args.device,optimizer,scheduler)
        validation = run_epoch(model,loaders[1],args.device)
        history.append({'epoch':epoch+1,'train':train,'validation':validation})
        (args.output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        # Atomic replace keeps the previous checkpoint if serialization fails.
        torch.save({'model_state':model.state_dict(),'optim_state':optimizer.state_dict(),
                    'lr_state':scheduler.state_dict(),'epoch':epoch+1,'config':config},args.output/'checkpoint.tmp')
        (args.output/'checkpoint.tmp').replace(args.output/'checkpoint.pt')
        print(history[-1])


if __name__ == '__main__':
    main()
