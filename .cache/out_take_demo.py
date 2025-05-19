"""
# README

Inspired by the paper "Transformer-VQ", it said that VQ on the attention Key so that we can get the linear attention easily, so I write this script.

## Test MSE

This function tests different quantization methods and compares their mean squared error (MSE) against the original attention output.

The result will be shown as a table, and we use some emoji here to make the result good-looking.

- heart 💛 means use the out-take trick, and different colors mean different algorithms.
- cross ❌ means some problem should be noted here, and it's common in our code.
- circle 🟡 means that do not out-take, and color same to heart.

When the data shape is small, like (3, 5, 4) and C = 3, the result is:

| 🟠mse1_1 | ❌mse1_2 | 🧡mse1_3 | 🟡mse2_1 | 🟡mse2_2 | ❌mse2_3 | 💛mse2_4 | 💛mse2_5 | 🟢mse3_1 | ❌mse3_2 | 💚mse3_3 |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| 0.229662 | 0.417231 | 0.315451 | 0.275816 | 0.275816 | 0.627587 | 0.57126  | 0.57126  | 0.317975 | 0.681158 | 0.802027 |

- Similar to origin attention: softmax_vq > argmax_vq > softmax_vq_outtake > argmax_vq_outtake

But it comes to the real condition, like (32, 2048, 128) and C = 64, the result is:

| 🟠mse1_1 | ❌mse1_2 | 🧡mse1_3 | 🟡mse2_1 | 🟡mse2_2 | ❌mse2_3 | 💛mse2_4 | 💛mse2_5 | 🟢mse3_1 | ❌mse3_2 | 💚mse3_3 |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| 0.74251  | 22.8576  | 1.24625  | 0.717983 | 0.717983 | 28.4505  | 1.48243  | 1.48243  | 0.717991 | 28.4145  | 1.48935  |

- Similar to origin attention: argmax_vq > softmax_vq > softmax_vq_outtake > argmax_vq_outtake

> Gumbel softmax is not good here.

## Test Performance

In the setting of (64, 4096, 512) and C = 512, we test the performance of different quantization methods.

We get the result here:

| Method                  | Cost Time (s) | Max Memory (MB) | Avg MSE   |
|-------------------------|---------------|-----------------|-----------|
| softmax_vq              | 63.8973       | 11785.1         | 1.06531   |
| softmax_vq_outtake      | 38.7384       | 10761.1         | 1.56572   |
| argmax_vq               | 62.9895       | 11275.1         | 0.961123  |
| argmax_vq_outtake       | 38.5751       | 10761.1         | 1.70298   |

> Cost time: argmax_vq_outtake < softmax_vq_outtake < argmax_vq < softmax_vq
> - Outtake time depends on C, but it is not good as we expected. When C 512->64, the time is not reduced as 1/8 (the time is not scalable).
> - Explain: this is what we want in out-take trick, we can get the linear attention.
> Similar to original: argmax_vq > softmax_vq > softmax_vq_outtake > argmax_vq_outtake
> - Softmax_vq is worse than argmax_vq when it came to the real condition. I think it is because the softmax not stable when the C is large (many values in the softmax dim make no sense but the max value make sense, so the softmax to aggregate the neighbor value make it not stable).

## Summary

Out-take trick is good, but it is not scalable when C is large. And the softmax is not stable when C is large, so the argmax is better than softmax in this condition.

When we just want to apply this trick to our implement later, we must re-think what we want, 

1. linear-attention and half of the time cost? out-take trick
2. small model? softmax_vq
3. large model? argmax_vq

"""
import torch
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm
import time
from typing import Callable

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_probs(query: torch.Tensor, key: torch.Tensor, transpose=False) -> torch.Tensor:
    """if transpose is true we do softmax on T dim, else on D dim"""
    qk = query @ key.transpose(-2, -1)
    return qk.transpose(-2, -1).softmax(-1) if transpose else qk.softmax(-1)

def get_idxs(query: torch.Tensor, key: torch.Tensor, transpose=False) -> torch.Tensor:
    """if transpose is true we do argmax on T dim, else on D dim"""
    qk = query @ key.transpose(-2, -1)
    return qk.transpose(-2, -1).argmax(-1) if transpose else qk.argmax(-1)

def get_one_hot(query: torch.Tensor, key: torch.Tensor, transpose=False) -> torch.Tensor:
    """use gumbel softmax"""
    qk = query @ key.transpose(-2, -1)
    return F.gumbel_softmax(qk.transpose(-2, -1) if transpose else qk, tau=0.2, hard=True)

def mse(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    return (tensor1 - tensor2).pow(2).mean().item()

def test_mse():
    # simple print condition
    # B, T, D = 3, 5, 4
    # C = 3

    # simple print condition
    B, T, D = 32, 2048, 128
    C = 64

    # real condition
    # B, T, D = 64, 4096, 512
    # C = 512

    metrics = []
    runs = 500

    pbar = tqdm(total=runs, desc="Running", dynamic_ncols=True)
    for _ in range(runs):
        query = torch.randn(B, T, D, device=device)
        key = torch.randn(B, T, D, device=device)
        value = torch.randn(B, T, D, device=device)
        codebook = torch.randn(C, D, device=device)
        a_orignal = get_probs(query, key) @ value
        
        # 1 softmax
        probs = get_probs(key, codebook)
        a_quantize1_1 = get_probs(query, probs @ codebook) @ value
        # 1_2 has same problem to 2_3
        a_quantize1_2 = get_probs(query, codebook) @ (probs.transpose(-2, -1) @ value)
        probs_T = get_probs(key, codebook, True)
        a_quantize1_3 = get_probs(query, codebook) @ (probs_T @ value)

        # 2 argmax
        # 2_1 is equal to 2_2
        idxs_codebook = get_idxs(key, codebook) # (B, T) idx in [C] codebook is (C, D)
        a_quantize2_1 = get_probs(query, codebook[idxs_codebook]) @ value
        idxs_onehot = F.one_hot(idxs_codebook, C).float()
        a_quantize2_2 = get_probs(query, idxs_onehot @ codebook) @ value
        
        # we should not apply transpose to idxs_onehot and matmul with value here
        # (idxs_onehot.transpose(-2, -1) @ value) is wrong
        # because we one hot are on the D dimension, not the T dimension
        a_quantize2_3 = get_probs(query, codebook) @ (idxs_onehot.transpose(-2, -1) @ value)

        # 2_4 is equal to 2_5
        idxs_kv = get_idxs(key, codebook, True) # (B C) idx in [T] value is (B,T,D)
        value_select = torch.stack([value[b, idxs_kv[b]] for b in range(B)])
        a_quantize2_4  = get_probs(query, codebook) @ value_select
        idxs_kv_onehot = F.one_hot(idxs_kv, T).float()
        a_quantize2_5  = get_probs(query, codebook) @ (idxs_kv_onehot @ value)

        # 3 gumbel softmax
        one_hot = get_one_hot(key, codebook)
        a_quantize3_1 = get_probs(query, one_hot @ codebook) @ value
        # 3_2 has same problem to 2_3
        a_quantize3_2 = get_probs(query, codebook) @ (one_hot.transpose(-2, -1) @ value)
        one_hot_T = get_one_hot(query, codebook, True)
        a_quantize3_3 = get_probs(query, codebook) @ (one_hot_T @ value)
        
        metric = {
            "🟠mse1_1": mse(a_orignal, a_quantize1_1),
            "❌mse1_2": mse(a_orignal, a_quantize1_2),
            "🧡mse1_3": mse(a_orignal, a_quantize1_3),
            "🟡mse2_1": mse(a_orignal, a_quantize2_1),
            "🟡mse2_2": mse(a_orignal, a_quantize2_2),
            "❌mse2_3": mse(a_orignal, a_quantize2_3),
            "💛mse2_4": mse(a_orignal, a_quantize2_4),
            "💛mse2_5": mse(a_orignal, a_quantize2_5),
            "🟢mse3_1": mse(a_orignal, a_quantize3_1),
            "❌mse3_2": mse(a_orignal, a_quantize3_2),
            "💚mse3_3": mse(a_orignal, a_quantize3_3),
        }
        metrics.append(metric)
        pbar.set_postfix(metric)
        pbar.update()
    pbar.close()
    
    print(tabulate(metrics, headers="keys", tablefmt="fancy_grid"))

    # get the average of the metrics
    avg = {k: sum([m[k] for m in metrics]) / runs for k in metrics[0]}
    print(tabulate([avg], headers="keys", tablefmt="fancy_grid"))


def measure_performance(func):
    def wrapper(*args, **kwargs):
        torch.cuda.empty_cache()
        # Record start time
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated(device)
        result = func(*args, **kwargs)
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated(device)
        total_time = end_time - start_time
        memory_used = end_memory - start_memory
        performance = {
            "function": func.__name__,
            "**total_time/second**": total_time,
            "memory_used/MB": memory_used / 1024**2,
            "start_memory/MB": start_memory / 1024**2,
            "end_memory/MB": end_memory / 1024**2,
        }
        print(tabulate([performance], headers="keys", tablefmt="fancy_grid"))
        print_cur_gpu_util()
        torch.cuda.empty_cache()
        return result
    return wrapper

def softmax_vq_attention(
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        codebook: torch.Tensor
    ) -> torch.Tensor:
    probs = get_probs(key, codebook)
    return get_probs(query, probs @ codebook) @ value    

def softmax_vq_outtake_attention(
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        codebook: torch.Tensor
    ) -> torch.Tensor:
    probs_T = get_probs(key, codebook, True)
    return get_probs(query, codebook) @ (probs_T @ value)

def argmax_vq_attention(
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        codebook: torch.Tensor
    ) -> torch.Tensor:
    idxs = get_idxs(key, codebook) # (B, T) idx in [C] codebook is (C, D)
    return get_probs(query, codebook[idxs]) @ value

def argmax_vq_outtake_attention(
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        codebook: torch.Tensor
    ) -> torch.Tensor:
    B, T, D = query.shape
    idxs_kv = get_idxs(key, codebook, True) # (B C) idx in [T] value is (B,T,D)
    idxs_kv_onehot = F.one_hot(idxs_kv, T).float()
    return get_probs(query, codebook) @ (idxs_kv_onehot @ value)

def print_cur_gpu_util():
    free_mem, total_mem = torch.cuda.mem_get_info(device)
    metric = {
        "memory_allocated/MB": torch.cuda.memory_allocated(device) / 1024**2,
        "**max_memory_allocated/MB**": torch.cuda.max_memory_allocated(device) / 1024**2,
        "memory_percentage/%": (total_mem - free_mem) / total_mem * 100,
        "**used_memory/GB**": (total_mem - free_mem) / 1024**3,
        "free_memory/GB": free_mem / 1024**3,
        "total_memory/GB": total_mem / 1024**3,
    }
    print(tabulate([metric], headers="keys", tablefmt="fancy_grid"))

@measure_performance
def test_func(attention_func: Callable):
    # real condition
    B, T, D = 64, 4096, 512
    C = 64
    runs = 200
    results = []
    pbar = tqdm(total=runs, desc="Running", dynamic_ncols=True)
    for _ in range(runs):
        query = torch.randn(B, T, D, device=device)
        key = torch.randn(B, T, D, device=device)
        value = torch.randn(B, T, D, device=device)
        codebook = torch.randn(C, D, device=device)
        a_orignal = get_probs(query, key) @ value
        a_quantize1_1 = attention_func(query, key, value, codebook)
        result = mse(a_orignal, a_quantize1_1)
        results.append(result)
        pbar.set_postfix({"mse": result})
        pbar.update()
    pbar.close()
    metric = {
        "function": attention_func.__name__,
        "average_mse": sum(results) / runs,
    }
    print(tabulate([metric], headers="keys", tablefmt="fancy_grid"))

def test_performance():
    # C = 512
    # test_func(softmax_vq_attention) # cost_time: 63.8973s max_mem: 11785.1MB avg_mse: 1.06531
    # test_func(softmax_vq_outtake_attention) # cost_time: 38.7384s max_mem: 10761.1MB avg_mse: 1.56572
    # test_func(argmax_vq_attention) # cost_time: 62.9895s max_mem: 11275.1MB avg_mse: 0.961123
    # test_func(argmax_vq_outtake_attention) # cost_time: 38.5751s max_mem: 10761.1MB avg_mse: 1.70298

    # C = 64
    # test_func(softmax_vq_attention) # cost_time: 60.7976 max_mem: 11336.2MB avg_mse: 0.860081
    # test_func(softmax_vq_outtake_attention) # cost_time: 32.0522s max_mem: 10760.2MB avg_mse: 1.59163
    # test_func(argmax_vq_attention) # cost_time: 60.9078s max_mem: 11274.2MB avg_mse: 0.8529
    test_func(argmax_vq_outtake_attention) # cost_time: 32.2828s max_mem: 10760.2MB avg_mse: 1.73461

if __name__ == "__main__":
    # test_mse()
    test_performance()
