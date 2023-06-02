import os
import torch
import wandb
from tqdm import tqdm 

from utils import setdir

class Trainer():
    """
    Training Helper Class
    """
    def __init__(self, conf, model, criterion, eval_func=None):
        
        self.max_epoch = conf.max_epochs
        self.batch_size = conf.batch_size
        self.max_len = conf.max_len
        self.lr = conf.lr
        
        self.init_model_parameters(model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1, T_mult=2, eta_min=0.001)
        
        self.save_dir = setdir(os.path.join(conf.save_dir, conf.ckpt_dir))
        self.save_period = conf.save_period
        self.criterion = criterion
        self.eval_func = eval_func
        wandb.init(project=conf.project_name)
        wandb.watch(self.model, self.criterion, log="all", log_freq=10)
       
        
    def init_model_parameters(self, model):
        self.model = model.cuda()
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
                
                
    def save(self, file_name='epoch_0'):
        file_name = f'{file_name}.pt'
        file_path = os.path.join(self.save_dir, file_name)
        torch.save(self.model.state_dict(), file_path)
        return file_path
    
    
    def train_one_epoch(self, dataloader, epoch):
        """
        one epoch training logic
        """
        self.model.train()
        iter_bar = tqdm(dataloader, desc='Train Iter (loss=X.XXX)')
        avg_loss = []
        for batch in iter_bar:
            iter_bar.desc
            batch = torch.stack([b.to("cuda") for b in batch])
            self.optimizer.zero_grad()
            
            pred, mu, std = self.model(batch)
            
            loss = self.criterion(pred.to("cuda"), batch, mu, std)
            avg_loss.append(loss.item())
            iter_bar.set_description('Train Iter (lr=%5.3f, loss=%5.3f)'%(self.optimizer.param_groups[0]["lr"], loss.item()))
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            torch.cuda.empty_cache()
            wandb.log({"epoch": epoch, "train loss": loss, "lr": self.optimizer.param_groups[0]["lr"] })
        wandb.log({"epoch": epoch, "train avg loss": sum(avg_loss)/len(avg_loss) })
        
    
    def train(self, train_loader, dev_loader = None):
        """
        Full training logic
        """
        for epoch in range(1, self.max_epoch + 1):
            self.train_one_epoch(train_loader, epoch)
            
            with torch.no_grad():
                if not dev_loader: continue 
                eval_loss, eval_acc = self.evaluate(dev_loader)
                wandb.log({"epoch": epoch, "eval loss": eval_loss, "eval ACC": eval_acc })
                
            if epoch % self.save_period == 0:
                file_name = f"epoch_{epoch}"
                self.save(file_name)

    
    def evaluate(self, dev_loader):
        """
        Evaluation Loop
        """
        self.model.eval()
        iter_bar = tqdm(dev_loader, desc='Eval Iter (loss=X.XXX)')
        
        losses, accuracies = [], []
        for i, batch in enumerate(iter_bar):
            batch = torch.stack([b.to("cuda") for b in batch])
            pred, mu, std = self.model(batch)
            pred = pred.to("cuda")
            loss = self.criterion(pred, batch, mu, std)
            
            losses.append(loss.item())

            acc = self.eval_func(pred, batch)
            accuracies.append(acc)
            iter_bar.set_description('Eval Iter (loss=%5.3f, acc=%5.3f)'%(loss.item(), acc.item()))
        
        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)
            