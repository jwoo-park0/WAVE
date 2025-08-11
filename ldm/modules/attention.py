from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import numpy as np 
from ldm.modules.diffusionmodules.util import checkpoint
from collections import defaultdict

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., use_softmax=True):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.use_softmax = use_softmax
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        
        
        h = self.heads

        
        q = self.to_q(x)  
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        if self.use_softmax:
            attn = sim.softmax(dim=-1)
        else: 
            attn = sim
            
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class RefAttention(nn.Module):
    
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        self.adaptive_range = None

    def forward(self, x, context=None, mask=None, **kwargs):

        h = self.heads

        input_ndim = x.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = x.shape
            x = x.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size, wh, channel = x.shape
            height = width = int(wh ** 0.5)
       
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q= rearrange(q, 'b n (h d) -> (b h) n d', h=h) 
        
        ex_out = torch.empty_like(q)
        if self.adaptive_range is None:
            assert self.adaptive_range is not None, "Adaptive range must be set before using RefAttention"


        for i, context_range in zip(range(batch_size), self.adaptive_range*2):
            half_range = (context_range - 1) // 2
            start_idx = i * h
            end_idx = start_idx + h
            
            curr_q = q[start_idx:end_idx]  
            if i < batch_size//2:
                
                context_indices = torch.arange(
                    max(0, i- half_range),
                    min(batch_size // 2 - 1, i+ half_range + 1)
                )
                while len(context_indices) < context_range:
                    if context_indices[0] > 0:  
                        context_indices = torch.cat((torch.tensor([context_indices[0] - 1]), context_indices))
                    if len(context_indices) < context_range and context_indices[-1] < batch_size // 2 - 1 :
       
                        context_indices = torch.cat((context_indices, torch.tensor([context_indices[-1] + 1])))
             
                curr_k = k.data[context_indices]
                curr_v = v.data[context_indices]
                if exists(mask):
                    mask1 = mask[context_indices]
                    idx = torch.where(context_indices==i)[0].item()
                    
            else:
                context_indices = torch.arange(max(batch_size //2, i-half_range),
                                               min(batch_size-1, i+half_range+1))
                while len(context_indices) < context_range:
                    if context_indices[0] > batch_size // 2:  
                        context_indices = torch.cat((torch.tensor([context_indices[0] - 1]), context_indices))
                    if len(context_indices) < context_range and context_indices[-1] < batch_size -1 :
                     
                        context_indices = torch.cat((context_indices, torch.tensor([context_indices[-1] + 1])))
                
                curr_k = k.data[context_indices]
                curr_v = v.data[context_indices]
                if exists(mask):
                    mask1 = mask[context_indices]
                    idx = torch.where(context_indices==i)[0].item()
        
            curr_k = curr_k.flatten(0,1).unsqueeze(0)  
            curr_v = curr_v.flatten(0,1).unsqueeze(0)  
            
            curr_k, curr_v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (curr_k, curr_v)) 
            
            sim = einsum('b i d, b j d -> b i j', curr_q, curr_k) * self.scale 

            if exists(mask):    
                
                max_neg_value = -torch.finfo(sim.dtype).max
                
                mask1 = F.interpolate(mask1.unsqueeze(1).float(), size=(sim.shape[1],sim.shape[1]),mode='nearest').squeeze().bool() # interpolation
                mask1[idx] = False  # self mask = 1's 
                expanded_mask = mask1.permute(1,2,0).reshape(sim.shape[1],-1)
                expanded_mask = expanded_mask.unsqueeze(0).repeat(h, 1, 1) #zero_mask 
                sim.masked_fill_(expanded_mask, max_neg_value)
                
            attn = sim.softmax(dim=-1)  
            out = einsum('b i j, b j d -> b i d', attn, curr_v)  
            ex_out[start_idx:end_idx] = out  

        output = ex_out
        output = rearrange(output, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(output)


class RefTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, use_softmax = True):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = RefAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                    context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout, use_softmax = use_softmax)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None,mask=None, **kwargs):
        inputs = (x, context, mask, kwargs)
        return checkpoint(self._forward_with_kwargs, (inputs,), self.parameters(), self.checkpoint)
    
    def _forward_with_kwargs(self, inputs):
        # Unpack inputs and kwargs
        x, context, mask, kwargs = inputs
        return self._forward(x, context, mask, **kwargs)
    
    def _forward(self, x, context=None, mask=None, **kwargs):
        #ipdb.set_trace()
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask, **kwargs) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, use_softmax=True):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                    context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout, use_softmax=use_softmax)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None):
        return checkpoint(self._forward, (x, context, mask), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, mask=None):
        
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_softmax=True):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                   disable_self_attn=disable_self_attn, use_softmax=use_softmax)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None, mask=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context, mask=mask)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in

class RefTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, cache_query_fn=None, use_softmax = True):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = n_heads
        self.dim_head = d_head
        self.depth = depth
        self.context_dim = context_dim
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.disable_self_attn = disable_self_attn
        self.cache_query_fn = cache_query_fn
        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [RefTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                   disable_self_attn=disable_self_attn, use_softmax = use_softmax)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None, mask=None, **kwargs):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context, mask=mask, **kwargs)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in

