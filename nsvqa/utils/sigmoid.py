import numpy as np

def calibrate_sigmoid(
    confidence: float,
    true_threshold=0.95,
    false_threshold=0.40,
    target_conf=0.60,
    target_prob=0.78,
    p_min=0.01,
    p_max=0.99,
    steepness_factor=0.7,
) -> float:
    """Map confidence to probability using a sigmoid function with adjustable steepness.

    Args:
        confidence: Input confidence score
        true_threshold: Upper threshold
        false_threshold: Lower threshold
        target_conf: Target confidence point
        target_prob: Target probability value
        p_min: Minimum probability
        p_max: Maximum probability
        steepness_factor: Controls curve steepness (0-1, lower = less steep)
    """
    if confidence <= false_threshold:
        return p_min

    if confidence >= true_threshold:
        return p_max

    # Calculate parameters to ensure target_conf maps to target_prob
    # For a sigmoid function: f(x) = L / (1 + e^(-k(x-x0)))

    # First, normalize the target point
    x_norm = (target_conf - false_threshold) / (true_threshold - false_threshold)
    y_norm = (target_prob - p_min) / (p_max - p_min)

    # Find x0 (midpoint) and k (steepness) to satisfy our target point
    x0 = 0.30  # Midpoint of normalized range

    # Calculate base k value to hit the target point
    base_k = -np.log(1 / y_norm - 1) / (x_norm - x0)

    # Apply steepness factor (lower = less steep)
    k = base_k * steepness_factor

    # With reduced steepness, we need to adjust x0 to still hit the target point
    # Solve for new x0: y = 1/(1+e^(-k(x-x0))) => x0 = x + ln(1/y-1)/k
    adjusted_x0 = x_norm + np.log(1 / y_norm - 1) / k

    # Apply the sigmoid with our calculated parameters
    x_scaled = (confidence - false_threshold) / (true_threshold - false_threshold)
    sigmoid_value = 1 / (1 + np.exp(-k * (x_scaled - adjusted_x0)))

    # Ensure we still hit exactly p_min and p_max at the thresholds
    # by rescaling the output slightly
    min_val = 1 / (1 + np.exp(-k * (0 - adjusted_x0)))
    max_val = 1 / (1 + np.exp(-k * (1 - adjusted_x0)))

    # Normalize the output
    normalized = (sigmoid_value - min_val) / (max_val - min_val)

    return p_min + normalized * (p_max - p_min)

