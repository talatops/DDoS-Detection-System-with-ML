/**
 * OpenCL kernel for parallel entropy computation
 * Computes entropy for multiple windows in parallel
 * Optimized for NVIDIA RTX 3060 (warp size 32)
 */

__kernel void compute_entropy(
    __global const uint* counts,        // Input: histogram counts [num_windows * num_bins]
    __global const uint* window_totals, // Input: total count per window [num_windows]
    __global float* entropy_out,        // Output: entropy values [num_windows]
    const uint num_bins                 // Number of bins per window
) {
    int window_id = get_global_id(0);
    int num_windows = get_global_size(0);
    
    if (window_id >= num_windows) return;
    
    uint total = window_totals[window_id];
    if (total == 0) {
        entropy_out[window_id] = 0.0f;
        return;
    }
    
    float entropy = 0.0f;
    const float log2 = 0.6931471805599453f; // log(2.0)
    
    // Compute entropy: H = -Σ p_i * log2(p_i)
    int base_idx = window_id * num_bins;
    for (int i = 0; i < num_bins; ++i) {
        uint count = counts[base_idx + i];
        if (count > 0) {
            float p = (float)count / (float)total;
            entropy -= p * native_log(p) / log2;  // Use native_log for speed
        }
    }
    
    entropy_out[window_id] = entropy;
}

/**
 * Compute entropy for multiple features in parallel
 * Each work-item processes one window and one feature type
 */
__kernel void compute_multi_entropy(
    __global const uint* counts,        // Input: histogram counts [num_windows * num_features * num_bins]
    __global const uint* window_totals, // Input: total count per window [num_windows]
    __global float* entropy_out,        // Output: entropy values [num_windows * num_features]
    const uint num_bins,                // Number of bins per feature
    const uint num_features             // Number of features (e.g., 6 for IP/port/size/protocol)
) {
    int gid = get_global_id(0);
    int num_windows = get_global_size(0) / num_features;
    int feature_id = gid % num_features;
    int window_id = gid / num_features;
    
    if (window_id >= num_windows) return;
    
    uint total = window_totals[window_id];
    if (total == 0) {
        entropy_out[gid] = 0.0f;
        return;
    }
    
    float entropy = 0.0f;
    const float log2 = 0.6931471805599453f;
    
    int base_idx = window_id * num_features * num_bins + feature_id * num_bins;
    for (int i = 0; i < num_bins; ++i) {
        uint count = counts[base_idx + i];
        if (count > 0) {
            float p = (float)count / (float)total;
            entropy -= p * native_log(p) / log2;
        }
    }
    
    entropy_out[gid] = entropy;
}

/**
 * Optimized entropy computation using local memory for reductions
 * Better for large number of bins
 */
__kernel void compute_entropy_local(
    __global const uint* counts,
    __global const uint* window_totals,
    __global float* entropy_out,
    const uint num_bins,
    __local float* local_entropy
) {
    int window_id = get_global_id(0);
    int local_id = get_local_id(0);
    int num_windows = get_global_size(0);
    
    if (window_id >= num_windows) return;
    
    uint total = window_totals[window_id];
    if (total == 0) {
        entropy_out[window_id] = 0.0f;
        return;
    }
    
    float entropy = 0.0f;
    const float log2 = 0.6931471805599453f;
    
    int base_idx = window_id * num_bins;
    int work_group_size = get_local_size(0);
    int bins_per_item = (num_bins + work_group_size - 1) / work_group_size;
    int start_bin = local_id * bins_per_item;
    int end_bin = min(start_bin + bins_per_item, (int)num_bins);
    
    // Each work-item processes a subset of bins
    for (int i = start_bin; i < end_bin; ++i) {
        uint count = counts[base_idx + i];
        if (count > 0) {
            float p = (float)count / (float)total;
            entropy -= p * native_log(p) / log2;
        }
    }
    
    local_entropy[local_id] = entropy;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduction in local memory
    for (int stride = work_group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            local_entropy[local_id] += local_entropy[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result
    if (local_id == 0) {
        entropy_out[window_id] = local_entropy[0];
    }
}

