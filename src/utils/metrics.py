import torch


class Metrics_k:
    def __init__(self,k):
        self.k=k
        self.sum={
            'hitrate':0.0,
            'mrr':0.0,
            'ndcg':0.0
        }
        self.n_samples=0
    
    def update(self,target,pred):
        pred = pred[:, :self.k]

        self.sum['hitrate']+=Metrics_k.hitrate_k(target,pred).item()
        self.sum['mrr']+=Metrics_k.mrr_k(target,pred).item()
        self.sum['ndcg']+=Metrics_k.ndcg_k(target,pred).item()
        self.n_samples+=target.size(0)


    def compute(self):
        answ={'hitrate':0.0,
              'mrr':0.0,
              'ndcg':0.0}
        for key in answ:
            answ[key]=self.sum[key]/self.n_samples
        return answ

    @staticmethod
    def get_rank(target:torch.Tensor,pred:torch.Tensor):
        # found, в каких случаях есть в топе, ranks - на какой позиции
        mask=(pred==target.unsqueeze(1))
        found=mask.any(dim=1)
        ranks=(mask.float().argmax(dim=1)+1)*found.float()
        return found,ranks
    
    @staticmethod
    def hitrate_k(target:torch.Tensor,pred:torch.Tensor):  
        found, _ =Metrics_k.get_rank(target,pred)
        return found.float().sum()
    
    @staticmethod
    def mrr_k(target:torch.Tensor,pred:torch.Tensor):  

        found,ranks=Metrics_k.get_rank(target,pred)
        if not found.any():
            return torch.tensor(0.0, device=pred.device)
        return (1/ranks[found]).sum()

    @staticmethod
    def ndcg_k(target:torch.Tensor,pred:torch.Tensor):  

        found,ranks=Metrics_k.get_rank(target,pred)
        if not found.any():
            return torch.tensor(0.0, device=pred.device)
        return (1/torch.log2(ranks[found]+1)).sum()

