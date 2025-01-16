import os, sys, argparse, datetime, random, numpy as np, torch, wandb
from omegaconf import OmegaConf
from models.model import MyModel
from loss import my_loss_func
from dataloaders import get_dataloaders

def main():
    # 1. Parse config + CLI overrides
    parser = argparse.ArgumentParser()
    parser.add_argument('--overrides', nargs='*', default=[])
    args = parser.parse_args()
    cfg = OmegaConf.load('config.yaml')
    for o in args.overrides:
        k,v = o.split('=')
        OmegaConf.update(cfg, k.split('.'), eval(v) if v.replace('.','',1).isdigit() else v)

    # 2. Ensure reproducibility
    seed = cfg.training.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    # 3. Setup (multi)GPU
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.distributed.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # 4. Build model, optim, sched
    model = MyModel(cfg).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.training.steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.mixed_precision)

    # 5. Dataloaders
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # 6. Logging (only rank=0)
    if local_rank == 0:
        with open('key.txt', 'r') as file:
            wandb_key = file.read().strip()
        
        wandb.login(key=wandb_key)
        tstamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        wandb.init(project=cfg.project_name, config=OmegaConf.to_container(cfg, resolve=True), name=tstamp)
        exp_dir = f"experiments/{tstamp}"
        os.makedirs(exp_dir, exist_ok=True)
        OmegaConf.save(cfg, f"{exp_dir}/config.yaml")

    # 7. Training loop (fixed steps)
    step = 0; train_iter = iter(train_loader)
    while step < cfg.training.steps:
        try: 
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        model.train(); opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg.training.mixed_precision):
            loss = my_loss_func(model(batch['input'].to(device)), batch['label'].to(device))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

        if local_rank == 0 and (step + 1) % cfg.training.log_freq == 0:
            model.eval()
            val_batch = next(iter(val_loader))
            with torch.no_grad():
                val_loss = my_loss_func(model(val_batch['input'].to(device)), val_batch['label'].to(device))
            wandb.log({
                'train_loss': loss.item(),
                'val_loss': val_loss.item(),
                'lr': opt.param_groups[0]['lr']
            }, step=step)

        step += 1

    # 8. Save final checkpoint (only rank=0)
    if local_rank == 0:
        torch.save(model.state_dict(), f"{exp_dir}/model.pt")

if __name__ == '__main__':
    main()
