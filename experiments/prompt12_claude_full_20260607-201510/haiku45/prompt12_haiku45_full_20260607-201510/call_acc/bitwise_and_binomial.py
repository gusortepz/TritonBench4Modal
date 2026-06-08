import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass


def bitwise_and_binomial(
    input: torch.Tensor,
    other: torch.Tensor,
    total_count: torch.Tensor,
    probs: Optional[torch.Tensor] = None,
    logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes bitwise AND of input and other, then applies Binomial distribution sampling.
    
    Args:
        input: First input tensor of integral or Boolean type.
        other: Second input tensor of integral or Boolean type.
        total_count: Number of Bernoulli trials, broadcastable with probs or logits.
        probs: Event probabilities (optional). One of probs or logits must be provided.
        logits: Event log-odds (optional). One of probs or logits must be provided.
    
    Returns:
        Sampled tensor from Binomial distribution.
    """
    # Ensure input and other are on the same device and have compatible dtypes
    if input.device != other.device:
        other = other.to(input.device)
    
    # Compute bitwise AND
    and_result = torch.bitwise_and(input, other)
    
    # Convert AND result to float for distribution sampling
    and_result_float = and_result.float()
    
    # Ensure total_count is broadcastable with and_result_float
    total_count_broadcasted = torch.broadcast_to(
        total_count.float(),
        torch.broadcast_shapes(and_result_float.shape, total_count.shape)
    )
    and_result_float = torch.broadcast_to(
        and_result_float,
        total_count_broadcasted.shape
    )
    
    # Handle probs and logits
    if probs is not None and logits is not None:
        raise ValueError("Only one of probs or logits should be provided")
    
    if probs is None and logits is None:
        raise ValueError("Either probs or logits must be provided")
    
    # Broadcast probs or logits to match the output shape
    if probs is not None:
        probs_broadcasted = torch.broadcast_to(
            probs.float(),
            total_count_broadcasted.shape
        )
        logits_broadcasted = None
    else:
        probs_broadcasted = None
        logits_broadcasted = torch.broadcast_to(
            logits.float(),
            total_count_broadcasted.shape
        )
    
    # Sample from Binomial distribution
    # torch.binomial uses (count, probs) signature
    if probs_broadcasted is not None:
        output = torch.binomial(
            total_count_broadcasted,
            probs_broadcasted
        )
    else:
        # Convert logits to probs
        probs_from_logits = torch.sigmoid(logits_broadcasted)
        output = torch.binomial(
            total_count_broadcasted,
            probs_from_logits
        )
    
    return output

##################################################################################################################################################



import torch
import torch.nn.functional as F

def test_bitwise_and_binomial():
    results = {}

    # Test case 1: Using `probs`
    input_tensor = torch.tensor([1, 0, 1, 0], dtype=torch.int32, device='cuda')
    other_tensor = torch.tensor([1, 1, 0, 0], dtype=torch.int32, device='cuda')
    total_count = torch.tensor([5, 5, 5, 5], dtype=torch.float32, device='cuda')
    probs = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')
    results["test_case_1"] = bitwise_and_binomial(input_tensor, other_tensor, total_count, probs=probs)

    # Test case 2: Using `logits`
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device='cuda')
    results["test_case_2"] = bitwise_and_binomial(input_tensor, other_tensor, total_count, logits=logits)

    # Test case 3: Different `total_count` with `probs`
    total_count_diff = torch.tensor([10, 10, 10, 10], dtype=torch.float32, device='cuda')
    results["test_case_3"] = bitwise_and_binomial(input_tensor, other_tensor, total_count_diff, probs=probs)

    # Test case 4: Different `total_count` with `logits`
    results["test_case_4"] = bitwise_and_binomial(input_tensor, other_tensor, total_count_diff, logits=logits)

    return results

test_results = test_bitwise_and_binomial()
