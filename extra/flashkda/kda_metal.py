"""
FlashKDA-Metal: Kimi Delta Attention forward pass on Apple Silicon

Ports the CUTLASS-based KDA kernels from MoonshotAI/FlashKDA to tinygrad's
Metal backend. The KDA mechanism uses gated delta attention with recurrent
state, enabling efficient linear attention on Apple Silicon GPUs.

Reference: https://github.com/MoonshotAI/FlashKDA
"""

from tinygrad import Tensor, dtypes
import math

def flash_kda_fwd(
    q: Tensor,        # [B, T, H, K] Query
    k: Tensor,        # [B, T, H, K] Key
    v: Tensor,        # [B, T, H, V] Value
    g: Tensor,        # [B, T, H, K] Gate (pre-activation)
    beta: Tensor,     # [B, T, H]    Beta logits (pre-sigmoid)
    scale: float,     # scalar scaling factor
    A_log: Tensor,    # [H]          Log-gate parameter
    dt_bias: Tensor,  # [H, K]       Gate bias
    lower_bound: float = -5.0,  # Gate lower bound
    initial_state: Tensor = None,  # [B, H, V, K] optional
    chunk_size: int = 64,  # chunk size for tiled computation
) -> tuple:
    """
    FlashKDA forward pass — Metal implementation via tinygrad.

    Implements gated delta attention with recurrent state tracking.
    The algorithm processes input in chunks for memory efficiency,
    similar to the CUTLASS tiled approach but using tinygrad ops
    that compile to Metal shaders.

    Args:
        q: Query tensor [B, T, H, K]
        k: Key tensor [B, T, H, K]
        v: Value tensor [B, T, H, V]
        g: Gate tensor [B, T, H, K] (pre-activation)
        beta: Beta logits [B, T, H] (sigmoid applied internally)
        scale: Scaling factor for attention
        A_log: Log-gate parameters [H]
        dt_bias: Gate bias [H, K]
        lower_bound: Gate lower bound clamp
        initial_state: Optional initial recurrent state [B, H, V, K]

    Returns:
        out: Output tensor [B, T, H, V]
        final_state: Final recurrent state [B, H, V, K]
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    # Apply scaling to queries
    q = q * scale

    # Compute gating: sigmoid(beta) for write gate
    beta_act = beta.sigmoid()  # [B, T, H]

    # Compute delta gate: softplus(g + dt_bias) clamped by lower_bound
    # A = exp(A_log) provides the base decay rate
    A = A_log.exp()  # [H]
    dt = (g + dt_bias.unsqueeze(0).unsqueeze(0)).softplus()  # [B, T, H, K]
    dt = dt.clip(min_=math.exp(lower_bound))

    # Decay factor per step
    decay = (A.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * dt).neg().exp()  # [B, T, H, K]

    # Initialize recurrent state
    if initial_state is not None:
        state = initial_state  # [B, H, V, K]
    else:
        state = Tensor.zeros(B, H, V, K, dtype=q.dtype)

    # Process in chunks for memory efficiency (tiled approach)
    num_chunks = (T + chunk_size - 1) // chunk_size
    outputs = []

    for chunk_idx in range(num_chunks):
        t_start = chunk_idx * chunk_size
        t_end = min(t_start + chunk_size, T)
        chunk_len = t_end - t_start

        # Extract chunk slices
        q_chunk = q[:, t_start:t_end]       # [B, C, H, K]
        k_chunk = k[:, t_start:t_end]       # [B, C, H, K]
        v_chunk = v[:, t_start:t_end]       # [B, C, H, V]
        beta_chunk = beta_act[:, t_start:t_end]  # [B, C, H]
        decay_chunk = decay[:, t_start:t_end]    # [B, C, H, K]

        # === Intra-chunk: causal attention within the chunk ===
        # QK^T with causal mask
        attn = q_chunk.permute(0, 2, 1, 3).matmul(
            k_chunk.permute(0, 2, 3, 1)
        )  # [B, H, C, C]

        # Apply causal mask
        causal_mask = Tensor.ones(chunk_len, chunk_len).tril()  # lower triangular
        attn = attn * causal_mask.unsqueeze(0).unsqueeze(0)

        # Compute intra-chunk output
        intra_out = attn.matmul(
            (v_chunk * beta_chunk.unsqueeze(-1)).permute(0, 2, 1, 3)
        )  # [B, H, C, V]

        # === Inter-chunk: contribution from recurrent state ===
        # Query against accumulated state
        inter_out = q_chunk.permute(0, 2, 1, 3).matmul(
            state.permute(0, 1, 3, 2)
        )  # [B, H, C, V]

        # Combine intra and inter chunk outputs
        chunk_out = (intra_out + inter_out).permute(0, 2, 1, 3)  # [B, C, H, V]
        outputs.append(chunk_out)

        # === Update recurrent state ===
        # Delta update: state = decay * state + beta * v^T * k
        for t in range(chunk_len):
            k_t = k_chunk[:, t:t+1]       # [B, 1, H, K]
            v_t = v_chunk[:, t:t+1]       # [B, 1, H, V]
            b_t = beta_chunk[:, t:t+1]    # [B, 1, H]
            d_t = decay_chunk[:, t:t+1]   # [B, 1, H, K]

            # Outer product: v^T * k -> [B, H, V, K]
            delta = (v_t * b_t.unsqueeze(-1)).permute(0, 2, 3, 1).matmul(
                k_t.permute(0, 2, 1, 3)
            )  # [B, H, V, K]

            # Apply decay and accumulate
            state = state * d_t.squeeze(1).unsqueeze(2) + delta

    # Concatenate all chunk outputs
    out = Tensor.cat(*outputs, dim=1)  # [B, T, H, V]

    return out, state


def flash_kda_fwd_reference(q, k, v, g, beta, scale, A_log, dt_bias, lower_bound=-5.0, initial_state=None):
    """
    Pure reference implementation for correctness testing.
    Processes token-by-token (slow but correct).
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    q = q * scale
    beta_act = beta.sigmoid()
    A = A_log.exp()
    dt = (g + dt_bias.unsqueeze(0).unsqueeze(0)).softplus()
    dt = dt.clip(min_=math.exp(lower_bound))
    decay = (A.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * dt).neg().exp()

    if initial_state is not None:
        state = initial_state
    else:
        state = Tensor.zeros(B, H, V, K, dtype=q.dtype)

    outputs = []
    for t in range(T):
        q_t = q[:, t]          # [B, H, K]
        k_t = k[:, t]          # [B, H, K]
        v_t = v[:, t]          # [B, H, V]
        b_t = beta_act[:, t]   # [B, H]
        d_t = decay[:, t]      # [B, H, K]

        # Decay state
        state = state * d_t.unsqueeze(2)

        # Delta update: state += beta * outer(v, k)
        delta = (v_t * b_t.unsqueeze(-1)).unsqueeze(-1) * k_t.unsqueeze(-2)
        state = state + delta.permute(0, 1, 2, 3)

        # Query state: out = q @ state^T
        out_t = (q_t.unsqueeze(-2) * state.permute(0, 1, 3, 2)).sum(-1)
        outputs.append(out_t.unsqueeze(1))

    out = Tensor.cat(*outputs, dim=1)
    return out, state
