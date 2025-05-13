import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist # For rank checking in prints
from timm.models.layers import drop_path # Assuming DropPath is used by your Attention/Block
import sys
from torch.utils.checkpoint import checkpoint # <<< --- ADD THIS IMPORT ---

# --- COPY THE ORIGINAL DropPath if needed by Block/Attention ---
# (Or ensure your main script imports it in a way that this file can see it,
#  but it's safer to include dependencies here or ensure they are globally available
#  before this patch is applied)
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

# --- YOUR MODIFIED Attention CLASS ---
class ModifiedAttention(nn.Module): # Give it a distinct name for clarity during patching
    # ... (Keep your ModifiedAttention class exactly as it is, no changes needed here) ...
    # ... (from your previous working version) ...
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None): 
        super().__init__()

        self._block_idx = "NOT_SET" # Initialize, will be set externally
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        sys.stdout.flush()
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn_scores_raw = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn_scores_raw = attn_scores_raw + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn_scores_raw = attn_scores_raw + rel_pos_bias
        
        is_ddp_initialized = dist.is_available() and dist.is_initialized()
        current_rank = dist.get_rank() if is_ddp_initialized else 0
        
        if current_rank == 0:
            block_id_for_print = self._block_idx if hasattr(self, '_block_idx') and self._block_idx is not None else "UnknownBlock"
            if isinstance(block_id_for_print, int) and block_id_for_print < 3 or \
               (attn_scores_raw.numel() > 0 and (torch.isnan(attn_scores_raw).any() or torch.isinf(attn_scores_raw).any() or attn_scores_raw.abs().max() > 30)): # Print for first 3 blocks or if problematic
                print(f"\nDEBUG EVA_VIT ATTN (Rank {current_rank}, Block {block_id_for_print}): BEFORE Softmax (Original Dtype)")
                print(f"  attn_scores_raw shape: {attn_scores_raw.shape}, dtype: {attn_scores_raw.dtype}")
                if attn_scores_raw.numel() > 0:
                    print(f"  attn_scores_raw min: {attn_scores_raw.min().item():.4f}, max: {attn_scores_raw.max().item():.4f}, mean: {attn_scores_raw.mean().item():.4f}")
                print(f"  attn_scores_raw has NaN: {torch.isnan(attn_scores_raw).any()}, has Inf: {torch.isinf(attn_scores_raw).any()}")
                if attn_scores_raw.numel() > 0 and attn_scores_raw.abs().max() > 30:
                     print(f"  WARNING: Large absolute values in attn_scores_raw: {attn_scores_raw.abs().max().item():.4f}")

        with torch.amp.autocast(device_type=attn_scores_raw.device.type, enabled=False):
            attn_fp32 = attn_scores_raw.to(torch.float32)
            
            if current_rank == 0:
                block_id_for_print = self._block_idx if hasattr(self, '_block_idx') and self._block_idx is not None else "UnknownBlock"
                if isinstance(block_id_for_print, int) and block_id_for_print < 3 or \
                   (attn_fp32.numel() > 0 and (torch.isnan(attn_fp32).any() or torch.isinf(attn_fp32).any() or attn_fp32.abs().max() > 30)):
                    print(f"DEBUG EVA_VIT ATTN (Rank {current_rank}, Block {block_id_for_print}): attn_fp32 (CASTED) BEFORE Softmax")
                    print(f"  attn_fp32 (casted) shape: {attn_fp32.shape}, dtype: {attn_fp32.dtype}")
                    if attn_fp32.numel() > 0:
                        print(f"  attn_fp32 (casted) min: {attn_fp32.min().item():.4f}, max: {attn_fp32.max().item():.4f}, mean: {attn_fp32.mean().item():.4f}")
                    print(f"  attn_fp32 (casted) has NaN: {torch.isnan(attn_fp32).any()}, has Inf: {torch.isinf(attn_fp32).any()}")

            softmax_output_fp32 = attn_fp32.softmax(dim=-1)

            if current_rank == 0:
                block_id_for_print = self._block_idx if hasattr(self, '_block_idx') and self._block_idx is not None else "UnknownBlock"
                if isinstance(block_id_for_print, int) and block_id_for_print < 3 or \
                   (softmax_output_fp32.numel() > 0 and (torch.isnan(softmax_output_fp32).any() or torch.isinf(softmax_output_fp32).any())):
                    print(f"DEBUG EVA_VIT ATTN (Rank {current_rank}, Block {block_id_for_print}): softmax_output_fp32 (FP32 Softmax Result)", flush=True)
                    print(f"  softmax_output_fp32 shape: {softmax_output_fp32.shape}, dtype: {softmax_output_fp32.dtype}", flush=True)
                    if softmax_output_fp32.numel() > 0:
                        print(f"  softmax_output_fp32 min: {softmax_output_fp32.min().item():.4e}, max: {softmax_output_fp32.max().item():.4e}, mean: {softmax_output_fp32.mean().item():.4e}", flush=True)
                    print(f"  softmax_output_fp32 has NaN: {torch.isnan(softmax_output_fp32).any()}, has Inf: {torch.isinf(softmax_output_fp32).any()}", flush=True)
                    if softmax_output_fp32.numel() > 0 and ((softmax_output_fp32 == 0).all() or (softmax_output_fp32 == 1).all() or ((softmax_output_fp32 > 0) & (softmax_output_fp32 < 1e-9)).all()):
                        print(f"  WARNING: softmax_output_fp32 shows extreme saturation.", flush=True)
            
            attn = softmax_output_fp32.to(q.dtype)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ModifiedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ModifiedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        from lavis.models.eva_vit import Mlp as OriginalMlp
        self.mlp = OriginalMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        # <<< --- ADD THIS --- >>>
        self.use_gradient_checkpointing = True # Default to True for ViT blocks

    # <<< --- ADD THIS HELPER METHOD --- >>>
    def _forward_impl(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

    # <<< --- MODIFY THIS FORWARD METHOD --- >>>
    def forward(self, x, rel_pos_bias=None):
        if self.training and self.use_gradient_checkpointing:
            # To align with the T5's likely default and the UserWarning, try use_reentrant=True first.
            # If the "marked as ready twice" error persists, you can try use_reentrant=False here.
            return checkpoint(self._forward_impl, x, rel_pos_bias, use_reentrant=True)
        else:
            return self._forward_impl(x, rel_pos_bias)