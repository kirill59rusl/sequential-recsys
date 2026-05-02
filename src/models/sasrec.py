import torch
import numpy as np
from tqdm import tqdm


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) 
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, item_num, args):
        super(SASRec, self).__init__()

        
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.hidden_units = args.hidden_units
        self.maxlen = args.maxlen
        
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen+1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):

            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(self, item_seqs,mask=None): 
        #из логов в скрытое признаковое пространство
        B,T=item_seqs.shape #B - batch ; T - time
        # эмбеддинги последовательностей товаров
        seqs = self.item_emb(torch.LongTensor(item_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5 #масштабирование

        #позиционные эмбеддинги
        poss = torch.arange(1, T + 1, device=self.dev).unsqueeze(0)
        poss = poss.repeat(B, 1)
        if mask is not None:
            poss*=mask.long()

        seqs += self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        attention_mask = ~torch.tril(torch.ones((T, T), dtype=torch.bool, device=self.dev))

        if mask is not None:
            key_padding_mask = ~mask  # инвертируем, т.к. True = игнорировать
        else:
            key_padding_mask = None
        
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1) # (T, B, C)

            if self.norm_first: #Pre-LayerNorm
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x,
                                                attn_mask=attention_mask,
                                                key_padding_mask=key_padding_mask)
                seqs = seqs + mha_outputs
                seqs = torch.transpose(seqs, 0, 1)
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs,
                                                attn_mask=attention_mask,
                                                key_padding_mask=key_padding_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = torch.transpose(seqs, 0, 1)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs) # (B, T, C) -

        return log_feats

    def forward(self, item_seqs, mask=None): # train  

        log_feats = self.log2feats(item_seqs,mask) 

        logits=log_feats.matmul(self.item_emb.weight.T)

        return logits 

    def predict(self, item_seqs, mask=None): # inference

        log_feats = self.log2feats(item_seqs, mask) 
        if mask is not None:
            lenghts=mask.sum(dim=1)
            batch_indices=torch.arange(len(lenghts),device=self.dev)
            last_indices=lenghts-1
            final_feat=log_feats[batch_indices,last_indices]# (B, I, C)
        else:
            final_feat = log_feats[:, -1, :] 
        
        logits = final_feat.matmul(self.item_emb.weight.T)

        return logits # preds # (B, item_num+1)
    
