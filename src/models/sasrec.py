import numpy as np
import torch


class PointWiseFF(torch.nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        # (B, T, C) -> (B, C, T)
        out = self.net(x.transpose(1, 2))
        return out.transpose(1, 2)
    

class SASRecBlock(torch.nn.Module): 
    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super().__init__()

        self.attention=torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True)
        
        self.forward_layer = PointWiseFF(hidden_dim, dropout_rate)

        self.layernorm1 = torch.nn.LayerNorm(hidden_dim)
        self.layernorm2 = torch.nn.LayerNorm(hidden_dim)
    

    def forward(self,x,attn_mask,padding):
        
    
        h=self.layernorm1(x)   

        attn_out,_=self.attention(h,h,h,attn_mask=attn_mask,key_padding_mask=padding,need_weights=False)   

        x = x + attn_out   
        x = x + self.forward_layer(self.layernorm2(x))
        
        return x


#class Args: hidden_dim, max_len, dropout, num_blocks 
        
class SASRec(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.hidden_dim = args.hidden_dim
        self.max_len = args.max_len
        
        self.item_emb = torch.nn.Embedding(
            args.num_items + 1,
            args.hidden_dim,
            padding_idx=0
        )

        self.pos_emb = torch.nn.Embedding(
            args.max_len,
            args.hidden_dim
        )

        self.dropout = torch.nn.Dropout(args.dropout)

        self.blocks=torch.nn.ModuleList([
            SASRecBlock(
                args.hidden_dim,
                args.num_heads,
                args.dropout
            )
            for _ in range(args.num_blocks)
        ])

        self.final_ln = torch.nn.LayerNorm(args.hidden_dim)

    def forward(self, item_seq, mask):

        B, T = item_seq.shape
        device = item_seq.device

        positions = torch.arange(
            T,
            device=device
        ).unsqueeze(0)

        x = self.item_emb(item_seq)
        x *= self.hidden_dim ** 0.5

        x = x + self.pos_emb(positions)

        x = self.dropout(x)

        attn_mask = torch.triu( #треугольная маска для внимания
            torch.ones(T, T, device=device),
            diagonal=1
        ).bool()

        
        padding_mask = ~mask

        for block in self.blocks:

            x = block(
                x,
                attn_mask,
                padding_mask
                )

        x = self.final_ln(x)

        logits = x @ self.item_emb.weight.T

        return logits

    @torch.no_grad()
    def predict( self, item_seq, mask):

        logits = self.forward(item_seq, mask)

        lengths = mask.sum(dim=1) - 1

        batch_idx = torch.arange(
            item_seq.size(0),
            device=item_seq.device
        )

        final_logits = logits[
            batch_idx,
            lengths
        ]

        return final_logits