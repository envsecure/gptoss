import torch
import torch.nn as nn


class Modelcfg:
    n_t_layers=8
    n_heads=4
    d_model=256
    vocab=50257
    context_size=256
    max_new_tokens=200

class SelfAttention(nn.Module):
    def __init__(self,d_model,d_out):
        super().__init__()
        self.w_q=nn.Linear(d_model,d_out)
        self.w_k=nn.Linear(d_model,d_out)
        self.w_v=nn.Linear(d_model,d_out)
        self.drop=nn.Dropout(0.1)

    def forward(self,x):
        
        q=self.w_q(x)
        k=self.w_k(x)
        v=self.w_v(x)
        d = q.size(-1)
        attn_scores = (q @ k.transpose(-2, -1)) / (d ** 0.5)
        B, t, _ = attn_scores.shape
        mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~mask, float("-inf")) #casual masking
        attn_mat=torch.softmax(attn_scores,dim=-1)
        attn_mat=self.drop(attn_mat)
        return attn_mat@v
    
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        d_heads=d_model//n_heads
        self.attention=nn.ModuleList(
            [
                SelfAttention(d_model,d_heads) for _ in range(n_heads)
            ]
        )
    def forward(self,x):
        temp=[head(x) for head in self.attention]
        return torch.cat(temp,-1)
    

class FeedForward(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.l1=nn.Linear(d_model,d_model*4)
        self.l2=nn.Linear(d_model*4,d_model)
    def forward(self,x):
        x=self.l1(x)
        x=torch.relu(x)
        x=self.l2(x)
        return x

class TransFormer(nn.Module):

    def __init__(self,d_model,n_heads):
        super().__init__()
        self.norm1=nn.RMSNorm(d_model)
        self.norm2=nn.RMSNorm(d_model)
        self.attn=MultiHeadAttention(d_model,n_heads)
        self.ff=FeedForward(d_model)
        self.resid_drop1 = nn.Dropout(0.1)
        self.resid_drop2 = nn.Dropout(0.1)

    def forward(self,x):
        temp=x
        x=self.norm1(x)
        x=self.attn(x)
        x=self.resid_drop1(x)
        temp=temp+x
        x=self.norm2(temp)
        x=self.ff(x)
        x=self.resid_drop2(x)
        return temp+x

class SLM(nn.Module):
    def __init__(self,cfg:Modelcfg):
        super().__init__()
        self.cfg=cfg
        self.val_embed=nn.Embedding(cfg.vocab,cfg.d_model)
        self.pos_embed=nn.Embedding(cfg.context_size,cfg.d_model)
        self.embed_drop = nn.Dropout(0.1)
        self.tflayers=nn.ModuleList([TransFormer(cfg.d_model,cfg.n_heads)for _ in range(cfg.n_t_layers)])
        self.fnorm=nn.RMSNorm(cfg.d_model)
        self.out_head=nn.Linear(cfg.d_model,cfg.vocab,bias=False)
        # Weight tying: share the token-embedding table with the output head (GPT-2 style)
        self.out_head.weight=self.val_embed.weight

    def forward(self,x):
        B, T = x.shape
        v_embed=self.val_embed(x)
        pos = torch.arange(0, T, device=x.device)
        pos_embed=self.pos_embed(pos)
        x=v_embed+pos_embed
        x=self.embed_drop(x)
        for layer in self.tflayers:
            x=layer(x)
        x=self.fnorm(x)
    
        return self.out_head(x)
    
    def predict(self,idx,max_new_tokens,top_k=None,temp=1,eos_id=50256):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_size:]
            with torch.no_grad():
                logits=self(idx_cond)
            logits = logits[:, -1, :]
            logits=logits/temp
            if top_k is not None:
                topk_vals, topk_idx = torch.topk(logits, top_k)
                filtered_logits = torch.full_like(logits, float('-inf'))
                filtered_logits.scatter_(1, topk_idx, topk_vals)
                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
            else :
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            if eos_id is not None and next_token.item() == eos_id:
                    idx = torch.cat((idx, next_token), dim=1)
                    break
            idx = torch.cat((idx, next_token), dim=1)

        return idx



    