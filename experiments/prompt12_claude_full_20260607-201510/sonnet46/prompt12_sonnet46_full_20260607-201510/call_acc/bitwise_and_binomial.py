import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple, Union
from torch import Tensor

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
    probs: torch.Tensor = None,
    logits: torch.Tensor = None,
) -> torch.Tensor:
    # Step 1: Compute bitwise AND of input and other
    and_result = torch.bitwise_and(input, other)

    # Step 2: Use the AND result as total_count for Binomial distribution
    # The and_result represents the number of trials
    # total_count parameter is also provided but the description says
    # "the result is used as input for the Binomial distribution"
    # with each element representing the number of trials
    # We interpret and_result as the total_count for Binomial

    # Convert and_result to float for use with torch.distributions.Binomial
    and_result_float = and_result.float()

    # Build Binomial distribution
    if probs is not None:
        dist = torch.distributions.Binomial(total_count=and_result_float, probs=probs)
    elif logits is not None:
        dist = torch.distributions.Binomial(total_count=and_result_float, logits=logits)
    else:
        # Default probability of 0.5 if neither is provided
        dist = torch.distributions.Binomial(total_count=and_result_float, probs=torch.tensor(0.5))

    # Step 3: Sample from the Binomial distribution
    sample = dist.sample()

    return sample

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
