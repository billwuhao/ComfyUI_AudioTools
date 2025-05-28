"""
This source code is modified by Shengkui Zhao based on https://github.com/lucidrains/FLASH-pytorch
"""
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from .conv_module import ConvModule
from .fsmn import UniDeepFsmn
from .layer_norm import CLayerNorm


# Helper functions

def identity(t, *args, **kwargs):
    """
    Returns the input tensor unchanged.

    Args:
        t (torch.Tensor): Input tensor.
        *args: Additional arguments (ignored).
        **kwargs: Additional keyword arguments (ignored).
        
    Returns:
        torch.Tensor: The input tensor.
    """
    return t

def append_dims(x, num_dims):
    """
    Adds additional dimensions to the input tensor.

    Args:
        x (torch.Tensor): Input tensor.
        num_dims (int): Number of dimensions to append.

    Returns:
        torch.Tensor: Tensor with appended dimensions.
    """
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))  # Reshape to append dimensions

def exists(val):
    """
    Checks if a value exists (is not None).

    Args:
        val: The value to check.

    Returns:
        bool: True if value exists, False otherwise.
    """
    return val is not None

def default(val, d):
    """
    Returns a default value if the given value does not exist.

    Args:
        val: The value to check.
        d: Default value to return if val does not exist.

    Returns:
        The original value if it exists, otherwise the default value.
    """
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    """
    Calculates the amount of padding needed to make a number a multiple of another.

    Args:
        n (int): The number to pad.
        mult (int): The multiple to match.

    Returns:
        int: The padding amount required to make n a multiple of mult.
    """
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder  # Return the required padding

# Scale Normalization class

class ScaleNorm(nn.Module):
    """
    ScaleNorm implements a scaled normalization technique for neural network layers.

    Attributes:
        dim (int): Dimension of the input features.
        eps (float): Small value to prevent division by zero.
        g (nn.Parameter): Learnable parameter for scaling.
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5  # Calculate scale factor
        self.eps = eps  # Set epsilon
        self.g = nn.Parameter(torch.ones(1))  # Initialize scaling parameter

    def forward(self, x):
        """
        Forward pass for the ScaleNorm layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Scaled and normalized output tensor.
        """
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale  # Compute norm
        return x / norm.clamp(min=self.eps) * self.g  # Normalize and scale

# Absolute positional encodings class

class ScaledSinuEmbedding(nn.Module):
    """
    ScaledSinuEmbedding provides sinusoidal positional encodings for inputs.

    Attributes:
        scale (nn.Parameter): Learnable scale factor for the embeddings.
        inv_freq (torch.Tensor): Inverse frequency used for sine and cosine calculations.
    """

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))  # Initialize scale
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # Calculate inverse frequency
        self.register_buffer('inv_freq', inv_freq)  # Register as a buffer

    def forward(self, x):
        """
        Forward pass for the ScaledSinuEmbedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Positional encoding tensor of shape (batch_size, sequence_length, dim).
        """
        n, device = x.shape[1], x.device  # Extract sequence length and device
        t = torch.arange(n, device=device).type_as(self.inv_freq)  # Create time steps
        sinu = einsum('i , j -> i j', t, self.inv_freq)  # Calculate sine and cosine embeddings
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)  # Concatenate sine and cosine embeddings
        return emb * self.scale  # Scale the embeddings

class OffsetScale(nn.Module):
    """
    OffsetScale applies learned offsets and scales to the input tensor.

    Attributes:
        gamma (nn.Parameter): Learnable scale parameter for each head.
        beta (nn.Parameter): Learnable offset parameter for each head.
    """

    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))  # Initialize scale parameters
        self.beta = nn.Parameter(torch.zeros(heads, dim))  # Initialize offset parameters
        nn.init.normal_(self.gamma, std=0.02)  # Normal initialization for gamma

    def forward(self, x):
        """
        Forward pass for the OffsetScale layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            List[torch.Tensor]: A list of tensors with applied offsets and scales for each head.
        """
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta  # Apply scaling and offsets
        return out.unbind(dim=-2)  # Unbind heads into a list

# Feed-Forward Convolutional Module

class FFConvM(nn.Module):
    """
    FFConvM is a feed-forward convolutional module with normalization and dropout.

    Attributes:
        dim_in (int): Input dimension of the features.
        dim_out (int): Output dimension after processing.
        norm_klass (nn.Module): Normalization class to be used.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass=nn.LayerNorm,
        dropout=0.1
    ):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),  # Normalize input
            nn.Linear(dim_in, dim_out),  # Linear transformation
            nn.SiLU(),  # Activation function
            ConvModule(dim_out),  # Convolution module
            nn.Dropout(dropout)  # Apply dropout
        )

    def forward(self, x):
        """
        Forward pass for the FFConvM module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing.
        """
        output = self.mdl(x)  # Pass through the model
        return output

class FFM(nn.Module):
    """
    FFM is a feed-forward module with normalization and dropout.

    Attributes:
        dim_in (int): Input dimension of the features.
        dim_out (int): Output dimension after processing.
        norm_klass (nn.Module): Normalization class to be used.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass=nn.LayerNorm,
        dropout=0.1
    ):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),  # Normalize input
            nn.Linear(dim_in, dim_out),  # Linear transformation
            nn.SiLU(),  # Activation function
            nn.Dropout(dropout)  # Apply dropout
        )

    def forward(self, x):
        """
        Forward pass for the FFM module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing.
        """
        output = self.mdl(x)  # Pass through the model
        return output

class FLASH_ShareA_FFConvM(nn.Module):
    """ 
    Fast Shared Dual Attention Mechanism with feed-forward convolutional blocks.
    Published in paper: "MossFormer: Pushing the Performance Limit of Monaural Speech Separation 
    using Gated Single-Head Transformer with Convolution-Augmented Joint Self-Attentions", ICASSP 2023.
    (https://arxiv.org/abs/2302.11824)
    
    Args:
        dim (int): Input dimension.
        group_size (int, optional): Size of groups for processing. Defaults to 256.
        query_key_dim (int, optional): Dimension of the query and key. Defaults to 128.
        expansion_factor (float, optional): Factor to expand the hidden dimension. Defaults to 1.
        causal (bool, optional): Whether to use causal masking. Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        rotary_pos_emb (optional): Rotary positional embeddings for attention. Defaults to None.
        norm_klass (callable, optional): Normalization class to use. Defaults to nn.LayerNorm.
        shift_tokens (bool, optional): Whether to shift tokens for attention calculation. Defaults to True.
    """
    
    def __init__(
        self,
        *,
        dim,
        group_size=256,
        query_key_dim=128,
        expansion_factor=1.,
        causal=False,
        dropout=0.1,
        rotary_pos_emb=None,
        norm_klass=nn.LayerNorm,
        shift_tokens=True
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)        
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        # Initialize positional embeddings, dropout, and projections
        self.rotary_pos_emb = rotary_pos_emb
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward layers
        self.to_hidden = FFConvM(
            dim_in=dim,
            dim_out=hidden_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        self.to_qk = FFConvM(
            dim_in=dim,
            dim_out=query_key_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        
        # Offset and scale for query and key
        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)

        self.to_out = FFConvM(
            dim_in=dim * 2,
            dim_out=dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        
        self.gateActivate = nn.Sigmoid() 

    def forward(self, x, *, mask=None):
        """
        Forward pass for FLASH layer.
        
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, features).
            mask (Tensor, optional): Mask for attention. Defaults to None.
        
        Returns:
            Tensor: Output tensor after applying attention and projections.
        """
        
        # Pre-normalization step
        normed_x = x 
        residual = x  # Save residual for skip connection

        # Token shifting if enabled
        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

        # Initial projections
        v, u = self.to_hidden(normed_x).chunk(2, dim=-1)
        qk = self.to_qk(normed_x)

        # Offset and scale
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u)

        # Output calculation with gating
        out = (att_u * v) * self.gateActivate(att_v * u)       
        x = x + self.to_out(out)  # Residual connection
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask=None):
        """
        Calculate attention output using quadratic and linear attention mechanisms.
        
        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, features).
            quad_q (Tensor): Quadratic query representation.
            lin_q (Tensor): Linear query representation.
            quad_k (Tensor): Quadratic key representation.
            lin_k (Tensor): Linear key representation.
            v (Tensor): Value representation.
            u (Tensor): Additional value representation.
            mask (Tensor, optional): Mask for attention. Defaults to None.
        
        Returns:
            Tuple[Tensor, Tensor]: Attention outputs for v and u.
        """
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        # Apply mask to linear keys if provided
        if exists(mask):
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # Rotate queries and keys with rotary positional embeddings
        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k))

        # Padding for group processing
        padding = padding_to_multiple_of(n, g)
        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: F.pad(t, (0, 0, 0, padding), value=0.), (quad_q, quad_k, lin_q, lin_k, v, u))
            mask = default(mask, torch.ones((b, n), device=device, dtype=torch.bool))
            mask = F.pad(mask, (0, padding), value=False)

        # Group along sequence for attention
        quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: rearrange(t, 'b (g n) d -> b g n d', n=self.group_size), (quad_q, quad_k, lin_q, lin_k, v, u))

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j=g)

        # Calculate quadratic attention output
        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g
        attn = F.relu(sim) ** 2  # ReLU activation
        attn = self.dropout(attn)

        # Apply mask to attention if provided
        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)

        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        # Calculate output from attention
        quad_out_v = einsum('... i j, ... j d -> ... i d', attn, v)
        quad_out_u = einsum('... i j, ... j d -> ... i d', attn, u)

        # Calculate linear attention output
        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g
            lin_kv = lin_kv.cumsum(dim=1)  # Cumulative sum for linear attention
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.)
            lin_out_v = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)

            lin_ku = einsum('b g n d, b g n e -> b g d e', lin_k, u) / g
            lin_ku = lin_ku.cumsum(dim=1)  # Cumulative sum for linear attention
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value=0.)
            lin_out_u = einsum('b g d e, b g n d -> b g n e', lin_ku, lin_q)
        else:
            lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / n
            lin_out_v = einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)

            lin_ku = einsum('b g n d, b g n e -> b d e', lin_k, u) / n
            lin_out_u = einsum('b g n d, b d e -> b g n e', lin_q, lin_ku)

        # Reshape and remove padding from outputs
        return map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out_v + lin_out_v, quad_out_u + lin_out_u))

class Gated_FSMN(nn.Module):
    """
    Gated Frequency Selective Memory Network (FSMN) class.
    
    This class implements a gated FSMN that combines two feedforward 
    convolutional networks with a frequency selective memory module.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        lorder (int): Order of the filter for FSMN.
        hidden_size (int): Number of hidden units in the network.
    """
    def __init__(self, in_channels, out_channels, lorder, hidden_size):
        super().__init__()
        # Feedforward network for the first branch (u)
        self.to_u = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        # Feedforward network for the second branch (v)
        self.to_v = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        # Frequency selective memory network
        self.fsmn = UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)

    def forward(self, x):
        """
        Forward pass for the Gated FSMN.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).
        
        Returns:
            Tensor: Output tensor after applying gated FSMN operations.
        """
        input = x
        x_u = self.to_u(x)  # Process input through the first branch
        x_v = self.to_v(x)  # Process input through the second branch
        x_u = self.fsmn(x_u)  # Apply FSMN to the output of the first branch
        x = x_v * x_u + input  # Combine outputs with the original input
        return x


class Gated_FSMN_Block(nn.Module):
    """
    A 1-D convolutional block that incorporates a gated FSMN.

    This block consists of two convolutional layers, followed by a 
    gated FSMN and normalization layers.
    
    Args:
        dim (int): Dimensionality of the input.
        inner_channels (int): Number of channels in the inner layers.
        group_size (int): Size of the groups for normalization.
        norm_type (str): Type of normalization to use ('scalenorm' or 'layernorm').
    """
    def __init__(self, dim, inner_channels=256, group_size=256, norm_type='scalenorm'):
        super(Gated_FSMN_Block, self).__init__()
        # Choose normalization class based on the provided type
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # First convolutional layer with PReLU activation
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)  # Normalization after first convolution
        self.gated_fsmn = Gated_FSMN(inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)  # Gated FSMN layer
        self.norm2 = CLayerNorm(inner_channels)  # Normalization after FSMN
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)  # Final convolutional layer

    def forward(self, input):
        """
        Forward pass for the Gated FSMN Block.
        
        Args:
            input (Tensor): Input tensor of shape (batch_size, dim, sequence_length).
        
        Returns:
            Tensor: Output tensor after processing through the block.
        """
        conv1 = self.conv1(input.transpose(2, 1))  # Apply first convolution
        norm1 = self.norm1(conv1)  # Apply normalization
        seq_out = self.gated_fsmn(norm1.transpose(2, 1))  # Apply gated FSMN
        norm2 = self.norm2(seq_out.transpose(2, 1))  # Apply second normalization
        conv2 = self.conv2(norm2)  # Apply final convolution
        return conv2.transpose(2, 1) + input  # Residual connection


class MossformerBlock_GFSMN(nn.Module):
    """
    Mossformer Block with Gated FSMN.

    This block combines attention mechanisms and gated FSMN layers 
    to process input sequences.
    
    Args:
        dim (int): Dimensionality of the input.
        depth (int): Number of layers in the block.
        group_size (int): Size of the groups for normalization.
        query_key_dim (int): Dimension of the query and key in attention.
        expansion_factor (float): Expansion factor for feedforward layers.
        causal (bool): If True, enables causal attention.
        attn_dropout (float): Dropout rate for attention layers.
        norm_type (str): Type of normalization to use ('scalenorm' or 'layernorm').
        shift_tokens (bool): If True, shifts tokens in the attention layer.
    """
    def __init__(self, *, dim, depth, group_size=256, query_key_dim=128, expansion_factor=4., causal=False, attn_dropout=0.1, norm_type='scalenorm', shift_tokens=True):
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # Rotary positional embedding for attention
        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))

        # Create a list of Gated FSMN blocks
        self.fsmn = nn.ModuleList([Gated_FSMN_Block(dim) for _ in range(depth)])

        # Create a list of attention layers using FLASH_ShareA_FFConvM
        self.layers = nn.ModuleList([
            FLASH_ShareA_FFConvM(
                dim=dim,
                group_size=group_size,
                query_key_dim=query_key_dim,
                expansion_factor=expansion_factor,
                causal=causal,
                dropout=attn_dropout,
                rotary_pos_emb=rotary_pos_emb,
                norm_klass=norm_klass,
                shift_tokens=shift_tokens
            ) for _ in range(depth)
        ])

    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        """
        Builds repeated UniDeep FSMN layers.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            lorder (int): Order of the filter for FSMN.
            hidden_size (int): Number of hidden units.
            repeats (int): Number of repetitions.
        
        Returns:
            Sequential: A sequential container with repeated layers.
        """
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for i in range(repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(self, x, *, mask=None):
        """
        Forward pass for the Mossformer Block with Gated FSMN.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, dim, sequence_length).
            mask (Tensor, optional): Mask tensor for attention operations.
        
        Returns:
            Tensor: Output tensor after processing through the block.
        """
        ii = 0
        for flash in self.layers:  # Process through each layer
            x = flash(x, mask=mask)
            x = self.fsmn[ii](x)  # Apply corresponding Gated FSMN block
            ii += 1
            
        return x


class MossformerBlock(nn.Module):
    """
    Mossformer Block with attention mechanisms.

    This block is designed to process input sequences using attention 
    layers and incorporates rotary positional embeddings. It allows 
    for configurable normalization types and can handle causal 
    attention.

    Args:
        dim (int): Dimensionality of the input.
        depth (int): Number of attention layers in the block.
        group_size (int, optional): Size of groups for normalization. Default is 256.
        query_key_dim (int, optional): Dimension of the query and key in attention. Default is 128.
        expansion_factor (float, optional): Expansion factor for feedforward layers. Default is 4.
        causal (bool, optional): If True, enables causal attention. Default is False.
        attn_dropout (float, optional): Dropout rate for attention layers. Default is 0.1.
        norm_type (str, optional): Type of normalization to use ('scalenorm' or 'layernorm'). Default is 'scalenorm'.
        shift_tokens (bool, optional): If True, shifts tokens in the attention layer. Default is True.
    """
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size=256,
        query_key_dim=128,
        expansion_factor=4.0,
        causal=False,
        attn_dropout=0.1,
        norm_type='scalenorm',
        shift_tokens=True
    ):
        super().__init__()

        # Ensure normalization type is valid
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        # Select normalization class based on the provided type
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size  # Group size for normalization

        # Rotary positional embedding for attention
        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))  
        # Max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J
        
        # Create a list of attention layers using FLASH_ShareA_FFConvM
        self.layers = nn.ModuleList([
            FLASH_ShareA_FFConvM(
                dim=dim,
                group_size=group_size,
                query_key_dim=query_key_dim,
                expansion_factor=expansion_factor,
                causal=causal,
                dropout=attn_dropout,
                rotary_pos_emb=rotary_pos_emb,
                norm_klass=norm_klass,
                shift_tokens=shift_tokens
            ) for _ in range(depth)
        ])

    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        """
        Builds repeated UniDeep FSMN layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            lorder (int): Order of the filter for FSMN.
            hidden_size (int): Number of hidden units.
            repeats (int, optional): Number of repetitions. Default is 1.

        Returns:
            Sequential: A sequential container with repeated layers.
        """
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for _ in range(repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(self, x, *, mask=None):
        """
        Forward pass for the Mossformer Block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, dim, sequence_length).
            mask (Tensor, optional): Mask tensor for attention operations.

        Returns:
            Tensor: Output tensor after processing through the block.
        """
        # Process input through each attention layer
        for flash in self.layers:
            x = flash(x, mask=mask)  # Apply attention layer with optional mask
        
        return x  # Return the final output tensor
