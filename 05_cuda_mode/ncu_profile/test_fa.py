import torch
from flash_attn import flash_attn_func


if __name__ == "__main__":
    
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    seqlen_q = seqlen_k = 32768
    d = 128
    nheads_k = nheads = 8
    dtype = torch.float16

    assert nheads % nheads_k == 0

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(batch_size, seqlen_k, nheads_k, d, device=device, dtype=dtype, requires_grad=False)

    with torch.no_grad():
        out, lse, S_dmask = flash_attn_func(q, k, v, return_attn_probs=True,)
    print(out)




