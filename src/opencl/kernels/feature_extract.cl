/**
 * OpenCL kernel for parallel feature extraction
 * Builds histograms and computes statistics in parallel
 */

/**
 * Build histogram for IP addresses (32-bit)
 * Each work-item processes one window
 */
__kernel void build_ip_histogram(
    __global const uint* src_ips,      // Input: source IPs [num_packets]
    __global const uint* window_offsets, // Input: window start indices [num_windows + 1]
    __global uint* histogram,          // Output: histogram [num_windows * histogram_size]
    const uint histogram_size,         // Size of histogram (number of bins)
    const uint num_windows              // Number of windows
) {
    int window_id = get_global_id(0);
    
    if (window_id >= num_windows) return;
    
    uint start_idx = window_offsets[window_id];
    uint end_idx = window_offsets[window_id + 1];
    
    int base_idx = window_id * histogram_size;
    
    // Initialize histogram to zero
    for (uint i = 0; i < histogram_size; ++i) {
        histogram[base_idx + i] = 0;
    }
    
    // Build histogram using hash-based binning
    for (uint i = start_idx; i < end_idx; ++i) {
        uint ip = src_ips[i];
        uint bin = ip % histogram_size;  // Simple hash
        histogram[base_idx + bin]++;
    }
}

/**
 * Build histogram for ports (16-bit)
 */
__kernel void build_port_histogram(
    __global const ushort* ports,
    __global const uint* window_offsets,
    __global uint* histogram,
    const uint histogram_size,
    const uint num_windows
) {
    int window_id = get_global_id(0);
    
    if (window_id >= num_windows) return;
    
    uint start_idx = window_offsets[window_id];
    uint end_idx = window_offsets[window_id + 1];
    
    int base_idx = window_id * histogram_size;
    
    // Initialize histogram
    for (uint i = 0; i < histogram_size; ++i) {
        histogram[base_idx + i] = 0;
    }
    
    // Build histogram
    for (uint i = start_idx; i < end_idx; ++i) {
        ushort port = ports[i];
        uint bin = port % histogram_size;
        histogram[base_idx + bin]++;
    }
}

/**
 * Compute packet size statistics
 */
__kernel void compute_packet_stats(
    __global const ushort* packet_sizes,
    __global const uint* window_offsets,
    __global float* stats_out,         // Output: [num_windows * 4] (min, max, mean, std)
    const uint num_windows
) {
    int window_id = get_global_id(0);
    
    if (window_id >= num_windows) return;
    
    uint start_idx = window_offsets[window_id];
    uint end_idx = window_offsets[window_id + 1];
    uint count = end_idx - start_idx;
    
    if (count == 0) {
        int base_idx = window_id * 4;
        stats_out[base_idx + 0] = 0.0f;  // min
        stats_out[base_idx + 1] = 0.0f;  // max
        stats_out[base_idx + 2] = 0.0f;  // mean
        stats_out[base_idx + 3] = 0.0f;  // std
        return;
    }
    
    // Compute min, max, sum
    ushort min_val = packet_sizes[start_idx];
    ushort max_val = packet_sizes[start_idx];
    uint sum = 0;
    
    for (uint i = start_idx; i < end_idx; ++i) {
        ushort size = packet_sizes[i];
        if (size < min_val) min_val = size;
        if (size > max_val) max_val = size;
        sum += size;
    }
    
    float mean = (float)sum / (float)count;
    
    // Compute variance
    float variance = 0.0f;
    for (uint i = start_idx; i < end_idx; ++i) {
        float diff = (float)packet_sizes[i] - mean;
        variance += diff * diff;
    }
    variance /= (float)count;
    float std_dev = sqrt(variance);
    
    // Write results
    int base_idx = window_id * 4;
    stats_out[base_idx + 0] = (float)min_val;
    stats_out[base_idx + 1] = (float)max_val;
    stats_out[base_idx + 2] = mean;
    stats_out[base_idx + 3] = std_dev;
}

/**
 * Count unique values using hash-based approach
 */
__kernel void count_unique(
    __global const uint* values,
    __global const uint* window_offsets,
    __global uint* unique_counts,
    const uint num_windows
) {
    int window_id = get_global_id(0);
    
    if (window_id >= num_windows) return;
    
    uint start_idx = window_offsets[window_id];
    uint end_idx = window_offsets[window_id + 1];
    
    // Simple approximation: count distinct hash values
    // For exact count, would need more complex data structures
    uint hash_set[256];  // Local hash table (simplified)
    uint hash_count = 0;
    
    for (uint i = 0; i < 256; ++i) {
        hash_set[i] = 0;
    }
    
    for (uint i = start_idx; i < end_idx; ++i) {
        uint value = values[i];
        uint hash = value % 256;
        
        if (hash_set[hash] == 0) {
            hash_set[hash] = value;
            hash_count++;
        } else if (hash_set[hash] != value) {
            // Collision - increment count anyway (approximation)
            hash_count++;
        }
    }
    
    unique_counts[window_id] = hash_count;
}

