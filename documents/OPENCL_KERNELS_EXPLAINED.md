## 1. High-level GPU & OpenCL overview

### 1.1 Why use GPUs for this project?
- **Massive parallelism**: A GPU can run thousands of tiny programs (called *work-items*) at the same time.
- **Our use case**: We repeatedly apply the same operations to many packets and many time windows:
  - Build histograms of IPs, ports, packet sizes, etc.
  - Compute statistics (min, max, mean, standard deviation).
  - Compute entropy values for different features.
- **Key idea**: Each GPU work-item can handle one *window* (or one window-feature pair), so we evaluate many windows in parallel. This is much faster than a CPU that processes windows mostly one-by-one.

### 1.2 OpenCL in simple terms
You can think of OpenCL as a way to:
- **Describe a small function** (a *kernel*) that runs on the GPU.
- **Send data** from the CPU (host) to GPU (device) in large arrays (OpenCL buffers).
- **Launch many copies** of that kernel in parallel, each copy working on different parts of the data.

Key concepts:
- **Device**: The GPU (e.g., NVIDIA RTX 3060).
- **Host**: The CPU that controls the GPU and the rest of the program.
- **Context**: Manages the device(s) and memory allocations.
- **Command queue**: The host uses it to send commands to the GPU (copy data, run kernels, etc.).
- **Kernel**: A function written in OpenCL C (like `compute_entropy`, `build_ip_histogram`).
- **Global work size**: How many parallel work-items we launch in total.
- **Local work size / work-group**: Groups of work-items that can cooperate using fast local memory and barriers.

### 1.3 Work-items and IDs
When we launch a kernel, OpenCL assigns each copy:
- A **global ID**: `get_global_id(0)` tells the work-item which element/window it should process.
- Optionally a **local ID** and **local size**: `get_local_id(0)`, `get_local_size(0)` tell it its position inside a work-group.

Our kernels use this pattern a lot:
- **One work-item per window** (e.g., computing entropy for each window).
- **One work-item per (window, feature) pair** (for multi-feature entropy).
- **One work-group per window** with many work-items collaborating (in the optimized entropy kernel with local memory).

### 1.4 Memory spaces we use
OpenCL distinguishes several memory spaces:
- `__global`: Large, slow memory accessible by all work-items and the host. We store:
  - Histograms.
  - Packet fields (IPs, ports, sizes).
  - Window offsets and totals.
  - Output arrays (entropy, statistics).
- `__local`: Fast memory shared inside a work-group, used for cooperation and reductions.
- *Private* variables: Normal local variables inside a kernel; each work-item has its own copy.

In this project:
- `__global` is used for all input and output buffers passed from the CPU.
- `__local` is used in the optimized entropy kernel to sum partial entropy contributions across work-items efficiently.

### 1.5 CPU vs GPU in this project
- **CPU side**:
  - Reads PCAPs, builds packet windows.
  - Prepares arrays for the GPU (IPs, ports, sizes, histograms, window offsets/totals).
  - Calls the OpenCL kernels.
  - Consumes GPU outputs (features and entropy) to feed ML models and detection logic.
- **GPU side**:
  - Quickly computes histograms and statistics per window.
  - Quickly computes entropy values per window (and per feature).
  - Returns compact feature vectors to the CPU.

The main reason for OpenCL here is to **speed up window-based feature extraction** for DDoS detection.

---

## 2. How this project uses OpenCL

### 2.1 Pipeline overview
At a high level, the DDoS detection pipeline is:
- Packets from PCAP / live traffic.
- Group packets into **time windows** or **fixed-size windows**.
- For each window, compute:
  - Histograms (IP, port, etc.).
  - Packet size statistics.
  - Unique counts, entropy.
- Feed these features into:
  - ML models (Random Forest, GBDT, DNN, etc.).
  - Rule-based or statistical detectors.

The OpenCL kernels in `src/opencl/kernels` are responsible for **the heavy per-window feature computation**:
- `feature_extract.cl`:
  - Builds histograms for IPs and ports.
  - Computes per-window packet size statistics.
  - Approximates unique value counts.
- `entropy.cl`:
  - Computes entropy from histograms, either:
    - One entropy per window, or
    - One entropy per (window, feature) pair.
  - Includes an optimized version using local memory.

### 2.2 Data sent to the GPU
The CPU prepares and sends to the GPU:
- **Packet fields arrays**:
  - `src_ips[i]`: Source IP of packet `i`.
  - `ports[i]`: Port of packet `i`.
  - `packet_sizes[i]`: Size of packet `i`.
- **Window metadata**:
  - `window_offsets[w]`: Starting packet index for window `w`.
  - `window_offsets[w+1]`: One past the last packet index for window `w`.
  - From these offsets, each kernel can find all packets belonging to a window.
- **Histograms & counts**:
  - Arrays that hold per-window histograms and total counts.
- **Output arrays**:
  - For each kernel, a global output buffer to store per-window results (entropy values, stats, unique counts).

The GPU kernels then **fill** or **transform** these arrays, and the CPU reads them back.

### 2.3 Design targets and assumptions
- The kernels are written to be **portable OpenCL**, but comments mention:
  - Optimization for **NVIDIA RTX 3060**, whose "warp size" is 32.
  - This matters mainly for performance tuning (how many threads cooperate in a group), not for correctness.
- Important assumptions in this design:
  - Each window is large enough that computing histograms and entropy is meaningful.
  - Histogram sizes and numbers of features are fixed or known at kernel launch time.
  - Data layout is flattened (1D arrays) for simplicity and performance.

---

## 3. Concepts needed to read the kernels

### 3.1 Work-item indexing patterns
You will see three main patterns:

1. **One work-item per window**  
   Example in entropy and histogram kernels:
   - `int window_id = get_global_id(0);`
   - `if (window_id >= num_windows) return;`
   - Each work-item handles exactly one logical window.

2. **One work-item per (window, feature) pair**  
   Used in the multi-entropy kernel:
   - Global ID `gid` enumerates all window/feature combinations.
   - `window_id = gid / num_features;`
   - `feature_id = gid % num_features;`

3. **One work-group per window, multiple work-items share the work**  
   Used in the optimized entropy kernel:
   - A group of work-items splits the bins for a window.
   - They sum partial entropy contributions into `__local` memory.
   - A reduction step combines everything into one entropy value.

### 3.2 Flattened array indexing
Instead of using multi-dimensional arrays, we use **1D arrays** and compute indices manually.

Typical patterns:
- **Per-window array with `num_bins` bins**:
  - Base index for window `w`: `base_idx = w * num_bins`.
  - Bin `i` in window `w`: `array[base_idx + i]`.

- **Per-(window, feature) array with `num_bins` bins per feature**:
  - Layout: `[window][feature][bin]` flattened into 1D.
  - Base index for `(window, feature)` pair:
    - `base_idx = window_id * num_features * num_bins + feature_id * num_bins;`
  - Bin `i`: `array[base_idx + i]`.

This layout is heavily used in `entropy.cl`.

### 3.3 Memory qualifiers
You will repeatedly see:
- `__global const uint*` / `__global uint*`: arrays in global memory (input or output).
- `__global float*`: output entropy or stats arrays.
- `__local float*`: local temporary storage for a work-group.

Understanding these qualifiers is enough to read and reason about the kernels in this project.

---

## 4. `feature_extract.cl` – feature-building kernels

This file contains kernels that transform raw packet fields into per-window features.

### 4.1 `build_ip_histogram`

**Purpose**: For each window, build a histogram over source IP addresses.
- Input:
  - `src_ips`: array of source IPs for all packets.
  - `window_offsets`: window start indices; `window_offsets[w]` and `window_offsets[w+1]` bound window `w`.
- Output:
  - `histogram`: for each window, a block of `histogram_size` bins.

Key logic (per work-item / window):
- Determine `window_id` from `get_global_id(0)`.
- Compute `start_idx` and `end_idx` for that window using `window_offsets`.
- Zero the window’s histogram slice.
- For each packet in the window:
  - Hash the IP into a bin: `bin = ip % histogram_size`.
  - Increment that bin.

In code (existing kernel for reference):

```17:36:src/opencl/kernels/feature_extract.cl
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
```

Intuition:
- We don’t store full IPs; we store how often packets fall into each bin.
- The modulo-based hash is simple and fast but can cause collisions (different IPs mapping to the same bin). For DDoS-style analysis, we mostly care about **distribution shape**, so this approximation is acceptable.

### 4.2 `build_port_histogram`

**Purpose**: Same idea as IP histogram, but for ports (16-bit).

Key steps:
- Each work-item selects its window via `window_id`.
- It zeroes a per-window histogram slice.
- It loops over packets in that window, hashes `ports[i]` into bins, increments counts.

The logic is almost identical to `build_ip_histogram`, but on `ushort ports` instead of `uint src_ips`.

### 4.3 `compute_packet_stats`

**Purpose**: For each window, compute:
- Minimum packet size.
- Maximum packet size.
- Mean (average) packet size.
- Standard deviation of packet sizes.

Inputs:
- `packet_sizes`: array of packet sizes.
- `window_offsets`: window boundaries.

Output:
- `stats_out`: per window, 4 floats: `[min, max, mean, std]`.

Mathematically, for a window with sizes \(x_1, \dots, x_n\):
- \( \text{min} = \min_i x_i \)
- \( \text{max} = \max_i x_i \)
- \( \text{mean} = \frac{1}{n} \sum_i x_i \)
- \( \text{variance} = \frac{1}{n} \sum_i (x_i - \text{mean})^2 \)
- \( \text{std} = \sqrt{\text{variance}} \)

In code (core part of the kernel):

```97:125:src/opencl/kernels/feature_extract.cl
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
```

Intuition:
- It scans the window twice:
  - First pass: compute min, max, and sum (for mean).
  - Second pass: compute variance around that mean.
- These basic statistics are useful for distinguishing normal vs attack traffic patterns.

### 4.4 `count_unique`

**Purpose**: Approximate the number of unique values per window (for some `values` array).

Approach:
- Uses a small fixed-size local hash table (`hash_set[256]` in private memory).
- Iterates over values in the window:
  - Computes a hash index `hash = value % 256`.
  - If that slot is empty, stores the value and increments `hash_count`.
  - If there is a different value already, it counts another “unique” as well (a very rough approximation).

Core logic:

```146:163:src/opencl/kernels/feature_extract.cl
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
```

Notes:
- This is **not an exact unique count**; collisions cause over-counting.
- It is fast and simple, which can be acceptable if we only need a rough sense of uniqueness.

---

## 5. `entropy.cl` – entropy kernels

This file computes Shannon entropy from histograms.

### 5.1 Shannon entropy recap

For a discrete distribution with probabilities \(p_i\), entropy is:
\[
H = -\sum_i p_i \log_2(p_i)
\]

Interpretation:
- High entropy: distribution is spread out (many bins with similar probabilities).
- Low entropy: distribution is concentrated (a few bins dominate).

In network traffic:
- **Normal traffic**: often has more diverse IPs, ports, and behaviors → higher entropy.
- **DDoS attack traffic**: may be highly concentrated (e.g., many packets from same source or to same target) → lower entropy in certain features.

The kernels here compute:
- Entropy per window from a histogram of counts.
- Optionally, entropy per (window, feature) for multiple feature types.

### 5.2 `compute_entropy`

**Purpose**: Compute entropy for each window from its histogram.

Inputs:
- `counts`: histogram counts, laid out as `[window][bin]` flattened (`num_windows * num_bins`).
- `window_totals`: total count (sum of all bins) for each window.
- `num_bins`: number of bins per window.

Output:
- `entropy_out[window]`: entropy value for each window.

Core loop:

```24:37:src/opencl/kernels/entropy.cl
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
```

Explanation:
- `p = count / total` approximates the empirical probability of each bin.
- `native_log(p)` computes the natural logarithm (base \(e\)) using a hardware-optimized function.
- To get base-2 logarithms we use:
  - \( \log_2(p) = \frac{\ln(p)}{\ln(2)} \)
  - Hence `native_log(p) / log2`.
- We accumulate `entropy -= p * log2(p)` exactly as in the formula.

### 5.3 `compute_multi_entropy`

**Purpose**: Compute entropy for **multiple features** per window in a single kernel launch.

Inputs:
- `counts`: histogram counts laid out as `[window][feature][bin]` flattened.
- `window_totals[window]`: total count per window (shared across features).
- `num_bins`: bins per feature.
- `num_features`: number of feature types.

Output:
- `entropy_out[gid]`: entropy per (window, feature).

Indexing:
- `gid = get_global_id(0)` runs over all window-feature pairs.
- Number of windows:
  - `num_windows = get_global_size(0) / num_features`.
- From `gid`:
  - `feature_id = gid % num_features`.
  - `window_id = gid / num_features`.

Core indexing for the counts:

```67:76:src/opencl/kernels/entropy.cl
    int base_idx = window_id * num_features * num_bins + feature_id * num_bins;
    for (int i = 0; i < num_bins; ++i) {
        uint count = counts[base_idx + i];
        if (count > 0) {
            float p = (float)count / (float)total;
            entropy -= p * native_log(p) / log2;
        }
    }
    
    entropy_out[gid] = entropy;
```

Intuition:
- This kernel lets us compute entropies of several different histograms (e.g., IP entropy, port entropy, packet size entropy, protocol entropy) at once.
- It uses the same entropy formula but with a 3D flattened layout.

### 5.4 `compute_entropy_local` – optimized version with local memory

**Purpose**: Compute entropy for each window, but more efficiently when `num_bins` is large.

Idea:
- Instead of one work-item looping over all bins, a whole work-group cooperates:
  - Each work-item processes a subset of bins.
  - Each computes a **partial entropy**.
  - They store these partial entropies in `__local` memory.
  - A reduction step sums them into a final entropy value for that window.

Key pieces:

```105:118:src/opencl/kernels/entropy.cl
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
```

Then the reduction:

```123:134:src/opencl/kernels/entropy.cl
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
```

Explanation:
- `bins_per_item` decides how many bins each work-item should handle.
- `local_entropy` is a shared array in fast local memory, one float per work-item.
- The reduction loop is a standard *tree reduction*:
  - First, half of the work-items add the other half.
  - Then a quarter add another quarter, etc.
  - At the end, `local_entropy[0]` holds the total entropy for that window.

This design takes advantage of:
- **Parallelism over bins** *within* each window.
- **Local memory and barriers** to cooperate efficiently among work-items.

---

## 6. Putting it all together in the detection pipeline

### 6.1 Data flow summary

1. **Ingest phase (CPU)**:
   - Read packets from PCAP or live capture.
   - Assign each packet to a **window** (e.g., based on time or packet count).
   - Build arrays:
     - `src_ips`, `ports`, `packet_sizes`, etc.
     - `window_offsets` to mark where each window starts and ends.

2. **Feature extraction (GPU: `feature_extract.cl`)**:
   - Launch `build_ip_histogram` and `build_port_histogram`:
     - Get per-window distributions of IPs and ports.
   - Launch `compute_packet_stats`:
     - Get per-window size statistics.
   - Optionally, launch `count_unique`:
     - Get a rough measure of uniqueness.

3. **Entropy computation (GPU: `entropy.cl`)**:
   - Using histograms and totals, compute:
     - Overall entropy per window (`compute_entropy`).
     - Or per-feature entropies (`compute_multi_entropy`).
     - Or use the optimized local-memory version for large histograms (`compute_entropy_local`).

4. **Detection / ML phase (CPU)**:
   - Read back GPU outputs (features and entropy).
   - Build feature vectors per window.
   - Run ML models and/or rule-based detectors to identify DDoS attacks.

### 6.2 Why entropy and histograms help with DDoS detection

- **Histograms of IPs and ports**:
  - In normal traffic, many IPs and ports might be involved.
  - In certain DDoS attacks, traffic may be dominated by a few IPs or target ports.

- **Entropy of distributions**:
  - High entropy suggests diverse, “random-looking” traffic.
  - Low entropy suggests concentration (e.g., many packets from/to the same endpoint).

- **Packet size statistics**:
  - Many DDoS attacks use characteristic packet sizes (e.g., many small packets).
  - Changes in min, max, mean, and std over time can be strong indicators.

By computing these features quickly on the GPU, the system can monitor many windows and detect anomalies in near real-time.

---

## 7. Practical notes, assumptions, and FAQ

### 7.1 Assumptions in the kernels

- **Histogram sizes**:
  - `histogram_size` must be chosen reasonably (not too small to avoid too many collisions, not too large to waste memory).
- **Windowing strategy**:
  - The code assumes windows with enough packets to make entropy and statistics meaningful.
- **Totals and counts**:
  - `window_totals[window_id]` must equal the sum of histogram counts for that window.
  - Some kernels return zero when there are no packets (`total == 0` or `count == 0`).

### 7.2 Performance considerations

- **Use of `native_log`**:
  - `native_log` is a device-specific, faster version of `log`. It may be slightly less accurate but is usually much faster.
  - For DDoS detection, approximate entropy is typically sufficient.

- **Local memory and reductions**:
  - `compute_entropy_local` uses `__local` memory and barriers for improved performance with large `num_bins`.
  - Choosing good work-group sizes (e.g., multiples of 32 on NVIDIA GPUs) can improve performance further.

- **Memory access patterns**:
  - Flattened arrays and sequential bin accesses are cache- and bandwidth-friendly.
  - Initializing histograms per window in a simple loop is easy to understand and safe, though there can be more advanced optimizations.

### 7.3 Short FAQ (for readers new to OpenCL)

1. *Do I need to understand all of OpenCL to modify these kernels?*  
   - No. For this project, it is enough to understand:
     - How `get_global_id` and indices map to windows and features.
     - How data is laid out in flattened arrays.
     - Basic memory qualifiers (`__global`, `__local`).

2. *Where is the kernel actually launched?*  
   - In the host C++ code (e.g., in `gpu_detector.cpp` / `host.cpp`), which:
     - Creates OpenCL buffers.
     - Sets kernel arguments.
     - Chooses global and local sizes.
     - Enqueues kernel execution on a command queue.

3. *Can I change the features we compute?*  
   - Yes, but you must:
     - Keep the data layouts consistent between host and kernels.
     - Update the ML feature-building code to expect the new / changed features.

4. *What happens if a window has no packets?*  
   - The kernels typically write zeros for entropy or stats, to avoid division by zero.

5. *Why do we use approximations (hashing, approximate unique counts)?*  
   - Exact methods can be too slow or memory-intensive.
   - For anomaly detection, approximate measures are often sufficient as long as they change significantly under attack conditions.

---

This document should give you enough background to:
- Understand, in plain language, what each OpenCL kernel in `entropy.cl` and `feature_extract.cl` does.
- See how they connect to the rest of the DDoS detection pipeline.
- Feel confident reading or lightly modifying these kernels without deep OpenCL experience.


