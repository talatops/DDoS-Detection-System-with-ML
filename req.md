Great — this is a solid, focused plan. Below I give a complete, step-by-step guide for a **single developer** building the **GPU (OpenCL) -based** analyzer using **CIC-DDoS2019 (primary)** + **CAIDA/MAWI (optional)**, with **Entropy-based** + **ML-based** detection and **RTBH simulation** for blocking. I include: system architecture (with ASCII diagrams), detailed development tasks, OpenCL kernel & host design notes, dataset handling, training & inference plan, blocking simulation design, metrics to collect, experiment scripts, and whether you need a UI (and a lightweight suggestion if you want one).

I’ll be concrete: filenames, commands, parameters, and places where pitfalls commonly appear.

---

# Quick snapshot — what you’ll build

1. **Ingest**: read pcap / flow CSVs and produce feature vectors in fixed-time windows (e.g., 1s and 5s).
2. **Feature extraction**: compute histograms & counts needed for entropy and ML features.
3. **GPU kernels (OpenCL)**: parallel entropy computation and batch inference (or parts of inference) on extracted feature batches.
4. **Local ML inference**: lightweight model (Random Forest or small NN). Optionally run inference on CPU for verification; run batched inference on GPU if you convert the model or implement a tiny NN in OpenCL.
5. **Decision & Alerting**: combine entropy & ML outputs (rule or weighted) → produce alerts.
6. **RTBH simulation**: controller receives alerts, writes blackhole list, and a local blocker module filters / drops flows in replay to simulate blocking.
7. **Metrics & Logging**: detection time, accuracy, throughput, latency, resource usage, blocking effectiveness.
8. **Experiment scripts**: automate replay, run detector, collect logs, and produce plots.

---

# Should you build a UI?

**Short answer:** **No, you don’t need a full UI** to satisfy the project. A clean CLI + automated plotting scripts (matplotlib / pandas) is sufficient and easier to reproduce.
**Optional (recommended for demo):** a **minimal web dashboard** (Flask) that shows live metrics (throughput, GPU utilization, alerts timeline) and lets you toggle blocking on/off. This helps demos but is *not required*.

---

# System architecture (component view)

```
           +---------------------+
           |  Packet Replay /    |   (tcpreplay reading .pcap)
           |  Ingest (tcpreplay) |----+
           +---------------------+    |
                                    v
        +------------------+    +-----------------+    +----------------+
        | Feature Extract  |--->| Batch Buffer /  |--->| GPU OpenCL     |
        | (1s/5s windows)  |    | Host-side Queue |    | Kernels:       |
        +------------------+    +-----------------+    | - Entropy     |
               |  |                                      | - (Opt) NN   |
               |  +------------------------------------->| inference     |
               |                                         +----------------+
               v                                                |
       +-----------------+                                      v
       | CPU-based ML    | <---(optional fallback)---+       +-----------------+
       | inference       |                             ----> | Decision &      |
       | (RandomForest)  |                                   | Alerting        |
       +-----------------+                                   +-----------------+
               |                                                  |
               |                                                  v
               |                                           +---------------+
               |                                           | RTBH Controller|
               |                                           | (blackhole list|
               |                                           |  manager)      |
               |                                           +-------+--------+
               |                                                   |
               v                                                   v
      +----------------------+                         +----------------------+
      | Metrics & Logging    |                         | Blocker (pcap filter |
      | (timestamps,cores,   |<------------------------| or netfilter)        |
      | GPU counters)        |  (logs alerts & effects) +----------------------+
      +----------------------+
```

---

# Sequence diagram — per-batch processing (ASCII)

```
tcpreplay --> FeatureExtractor : produce window batch (1s)
FeatureExtractor -> HostQueue : enqueue features
HostQueue -> OpenCL Host : copy batch to device buffer (async)
OpenCL Host -> GPU Kernel : run entropy kernel (parallel)
GPU Kernel --> OpenCL Host : return entropy values
OpenCL Host -> ML Inference (CPU or GPU) : run model on batch
ML -> Decision : compute combined score (entropy weight + ML score)
Decision -> Alert Logger : if score > threshold emit alert
Alert Logger -> RTBH Controller : send IPs to blackhole list
RTBH Controller -> Blocker : instruct to drop matching packets
MetricsLogger <- Decision, Blocker : log detection time, dropped %
```

---

# Development roadmap (single developer) — concrete milestones

**Estimated total**: 6–8 weeks of dedicated work (adjust to your schedule). Each milestone contains concrete deliverables and suggested filenames.

1. **Setup & dataset prep (Days 1–3)**

   * Install tools: `tcpreplay`, `tshark`, `scapy`, OpenCL ICD & drivers, `clang`/`gcc`, Python (3.9+), `numpy`, `pandas`, `scikit-learn`, `matplotlib`.
   * Download CIC-DDoS2019 + CAIDA/MAWI pcaps or flow CSVs.
   * Deliverable: `data/README.md` with dataset sources and a script `scripts/prepare_data.py` that extracts relevant pcap segments and labels.

2. **Baseline ingestion & CPU feature extraction (Days 4–7)**

   * Implement packet parsing & windowing script: `src/ingest/replay_ingest.py` (or C).
   * Output a CSV per window containing counts/histograms. Example fields: `window_start, src_ip, dst_ip, src_port, dst_port, pkt_count, byte_count`.
   * Deliverable: `tools/pcap_to_windowed_csv.py`.

3. **Entropy detector CPU baseline (Days 8–10)**

   * Implement entropy calculation on CPU per-window. `src/detectors/entropy_cpu.py`.
   * Validate: small pcap with known attack should show entropy drop/increase. Plot entropy over time.

4. **ML pipeline — offline training (Days 11–16)**

   * Choose features (see below). Train RandomForest (scikit-learn) and export model: `models/rf_model.joblib`.
   * Scripts: `src/ml/train_ml.py`, `src/ml/features_spec.json`.
   * Evaluate offline accuracy and generate ROC/PR curves.

5. **OpenCL host + kernel scaffolding (Days 17–23)**

   * Create host skeleton: `src/opencl/host.c` (or C++). Implement device selection, buffer management, event profiling.
   * Write kernels: `src/opencl/kernels/entropy.cl`.
   * Test correctness by comparing OpenCL output vs CPU for same window.

6. **Integrate ML inference (Days 24–28)**

   * Option A (simple): keep ML inference on CPU, but run batched classification — still meets “offload at least one major computation to GPU” since entropy is on GPU.
   * Option B (advanced): implement tiny NN inference in OpenCL or convert a tiny NN to OpenCL (more work).
   * Deliverable: `src/opencl/host_runner.c` that: reads batch, runs OpenCL kernel(s), calls CPU inference, makes decision.

7. **RTBH simulation & blocker (Days 29–33)**

   * Implement RTBH controller: `src/blocking/rtbh_controller.py` — maintains blackhole list and writes actions.
   * Implement blocker: `src/blocking/pcap_filter.py` using scapy to drop packets from blackholed IPs during replay, or use `tcpreplay` + `--multiplier` with prefiltered pcap.
   * Alternative: integrate with `iptables` to drop traffic in a test VM. For reproducibility, prefer pcap filtering.

8. **Performance instrumentation (Days 34–38)**

   * Implement timers for: ingestion → feature extraction → kernel launch → kernel finish → inference → decision.
   * Log GPU kernel times via OpenCL event profiling (`clGetEventProfilingInfo`).
   * Collect CPU and memory via `psutil`. For GPU usage vendor tools are optional (e.g., AMD/Intel/NVIDIA vendor tools).

9. **Experiments & plots (Days 39–45)**

   * Scripts to run multiple experiments: `scripts/run_experiment.sh` with arguments: `--pps 100k --window 1 --model rf`.
   * Generate plots: detection lead time vs rate, throughput vs batch size, TPR/FPR, blocking effectiveness.

10. **Report & slides (Days 46–50)**

    * Assemble results, diagrams, and write the CCP justification. Prepare 10-12 slides.

---

# Feature choices & ML specifics (concrete)

**Windowing**

* Use **two parallel windows**: `1-second` window for low lead time detection, and `5-second` for stability analysis. Windows are sliding or tumbling? Use **tumbling** windows for simplicity.

**Entropy features (per window)**

* Source IP entropy (H_src)
* Destination IP entropy (H_dst)
* Source port entropy (H_sport)
* Packet size entropy (H_len)
* Protocol distribution entropy

Entropy formula:

```
H = - Σ p_i * log2(p_i)
```

Compute p_i = count_i / total_count for categories (IPs/ports/bins).

**Other features (for ML)**

* Total packets, total bytes
* Unique src IP count, unique dst IP count
* Top-N src IP fraction (e.g., fraction of traffic from top 10 IPs)
* Average packet size
* Flow count (5-tuple flows)
* Entropy features above

**ML model**

* **RandomForest** (scikit-learn) with `n_estimators=100, max_depth=10`. Save model with joblib.
* Train on aggregated windows labeled by CIC dataset attack labels.
* Evaluate via 5-fold cross validation: precision, recall, F1.

**Combining Entropy + ML**

* Two simple designs:

  1. **Ensemble rule**: alert if (`entropy_score` > E_thresh) OR (`RF_score` > RF_thresh)
  2. **Weighted score**: `score = w_e * normalized_entropy_score + w_m * RF_prob` → threshold
* For simplicity: use ensemble rule; report both detectors’ independent and combined performance.

---

# OpenCL design & kernel notes

**Which computations to offload?**

* Entropy computation is a great candidate: histogram building and computing -Σp log p across many categories can be parallelized across bins or keys.
* Optionally implement histogram accumulation on GPU for many src IP addresses (needs mapping).

**Data layout**

* Input: arrays of features per packet or pre-aggregated per-window histograms.
* Best practice: **do as much aggregation on CPU as makes sense** (group by IP into bins) and offload the heavy math: many log and multiply ops for many bins across many windows/batches.

**Kernel example idea (entropy over M bins for B windows)**

`entropy.cl` pseudo-kernel:

```c
__kernel void compute_entropy(__global const uint *counts, // size B*M
                              __global const uint *window_totals, // size B
                              __global float *entropy_out) {     // size B
    int window = get_global_id(0); // one work-item per window
    int M = ...; // compile-time or passed
    uint total = window_totals[window];
    float H = 0.0f;
    for (int i=0; i<M; ++i) {
        uint c = counts[window*M + i];
        if (c > 0) {
            float p = (float)c / (float)total;
            H += -p * log(p); // natural log or log2 (consistent with CPU)
        }
    }
    entropy_out[window] = H;
}
```

**Note:** The above loops inside a work-item are okay if M small (e.g., 256 bins). If M large, consider splitting over local work-groups and using reductions.

**Host flow (OpenCL)**

1. Create context & command queue with profiling enabled.
2. Allocate device buffers for counts, totals, and outputs.
3. `clEnqueueWriteBuffer` counts & totals (use pinned memory if available).
4. `clEnqueueNDRangeKernel` with `global_work_size = num_windows`.
5. `clEnqueueReadBuffer` entropy results (or read asynchronously and use events).
6. Measure kernel time with event profiling: `clGetEventProfilingInfo(CL_PROFILING_COMMAND_START/END)`.
7. Release buffers.

**Performance tips**

* Minimize host-device transfers — batch many windows (e.g., batches of 256 windows).
* Use `clEnqueueWriteBuffer` with `CL_FALSE` and `clFlush`/`clFinish` appropriately to pipeline transfers.
* Use local memory for reductions if doing per-work-group sums.
* Check device-specific alignment and preferred work-group sizes.

---

# Blocking — RTBH simulation (design)

**RTBH real idea**: mark route to blackhole so traffic to victim is dropped. In your simulation (single-developer, local) implement:

1. **Blackhole list**: a central list of IPs to block (or networks).
2. **Blocker implementation options**:

   * **pcap filtering (recommended)**: when replaying, use a dynamic filter that drops packets matching blackhole IPs. Implementation: `blocker.py` listens on a control channel (e.g., simple socket) for updated blackhole list and filters packets as they are replayed (use scapy `sniff()` & `send()` with drop).
   * **Prefiltered pcap**: when blackhole list updates, regenerate a filtered pcap (expensive).
   * **iptables**: if you have root access in environment and a test VM, add `iptables -A INPUT -s <ip> -j DROP` rules. Less reproducible; will interfere with host.

**Concrete blocker (pcap filter) approach**

* Run `tcpreplay` with `--intf1=lo` to loopback or use custom scapy replay that queries the blackhole set for each packet to decide to send.
* `rtbh_controller.py` writes a JSON file `blackhole.json` when new IPs are added. `pcap_filter.py` checks this file (or subscribes via a simple REST API) to drop matching packets.

**Metrics to produce for blocking**

* `% of attack packets dropped` after RTBH rule applied.
* Collateral damage: `% of legitimate packets dropped` (must label windows as attack vs benign).

---

# Experiment scripts & reproducibility

**Primary script**
`scripts/run_experiment.sh` (example usage)

```bash
# args: pcap, pps, window_secs, batch_size, model
./scripts/run_experiment.sh data/attack.pcap 200000 1 128 rf
```

**What the script does**

1. Start metrics logger: `python tools/metrics_collector.py --out logs/exp1/`.
2. Start blocker (listens for rules).
3. Start detector: OpenCL host binary `bin/detector --model models/rf_model.joblib --window 1 --batch 128`.
4. Start replay: `tcpreplay --pps 200000 -i lo data/attack.pcap`.
5. After replay finishes, gracefully stop detector and blocker, collect logs, run `scripts/plot_results.py logs/exp1`.

**Logging format**

* CSV logs for events: `alerts.csv` (timestamp_ms, window_start_ms, src_ip, score, detector).
* `metrics.csv` (timestamp_ms, cpu%, gpu%, memMB, pps_in, pps_processed).
* `blocking.csv` (timestamp, blackhole_applied, impacted_packets, dropped_packets).

---

# Evaluation plan (experiments to run — minimum)

1. **Correctness**

   * Compare OpenCL entropy vs CPU entropy for same windows.
   * Compare ML inference vs scikit-learn baseline.

2. **Detection accuracy**

   * Run detector over labeled dataset → compute Precision, Recall, F1, FPR (per-window labels).

3. **Lead Time**

   * For each attack in dataset: compute time between labeled attack start and first alert.

4. **Throughput**

   * Vary replay rates (100k pps, 200k, 500k, 1M) and measure processed pps and kernel throughput. Report Gbps.

5. **Latency**

   * Log per-window processing latency: average & 95th percentile.

6. **Resource utilization**

   * Plot CPU% and GPU kernel occupancy timeline.

7. **Blocking effectiveness**

   * Enable RTBH after threshold and measure % attack traffic dropped and % collateral damage.

8. **Ablation**

   * Entropy-only vs ML-only vs combined.

---

# Code structure suggestion (repository layout)

```
project-root/
├─ data/
├─ src/
│  ├─ ingest/
│  │  └─ pcap_reader.py
│  ├─ opencl/
│  │  ├─ host.c
│  │  └─ kernels/
│  ├─ detectors/
│  │  ├─ entropy_cpu.py
│  │  └─ detector_main.c
│  ├─ ml/
│  │  ├─ train_ml.py
│  │  └─ model_utils.py
│  ├─ blocking/
│  │  ├─ rtbh_controller.py
│  │  └─ pcap_filter.py
│  └─ utils/
│     └─ metrics_collector.py
├─ scripts/
│  └─ run_experiment.sh
├─ models/
└─ report/
```

---

# Practical commands & small examples

**Install tools (Ubuntu example)**

```bash
sudo apt update
sudo apt install -y tcpreplay tshark python3-pip build-essential ocl-icd-opencl-dev
pip3 install numpy pandas scikit-learn matplotlib psutil joblib scapy
```

**Run tcpreplay at specific pps**

```bash
tcpreplay --pps=200000 --intf1=lo data/attack.pcap
```

**OpenCL profiling (host snippet, pseudo-C)**

```c
cl_event event;
clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &event);
clWaitForEvents(1, &event);
cl_ulong start, end;
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
double time_ms = (end - start) * 1e-6;
```

---

# Common pitfalls & debugging tips

* **Start small**: validate with tiny pcaps (10s) before scaling to high pps.
* **Time synchronization**: use timestamps from replay or embed relative times in logs. Avoid relying on absolute system clocks across containers.
* **Host-device transfer costs**: batch windows to amortize transfer overhead.
* **Kernel correctness**: check floating-point difference (use tolerances).
* **Overfitting ML**: hold-out test set from CIC dataset; optionally test on CAIDA/MAWI background traces to measure false positives.
* **Blocker delays**: account for time between alert and blackhole application in lead time calculations.

---

# What to include in your report (10–12 pages) — tailored to this plan

* Abstract & CCP justification (2 paragraphs)
* Datasets & pre-processing (CIC + CAIDA/MAWI)
* System architecture (diagram + modules)
* Detection algorithms (entropy math + RF hyperparameters)
* OpenCL kernel design & host pipeline
* RTBH simulation architecture
* Experimental setup (hardware, replay rates, window sizes)
* Results (tables + plots for all required metrics)
* Discussion (trade-offs, errors, limitations)
* Conclusion & future work
* Appendices: commands to reproduce, code structure

---

# Final checklist (immediately actionable)

* [ ] Install tools & drivers; verify OpenCL device visible (`clinfo`).
* [ ] Download CIC-DDoS2019; make small test pcap.
* [ ] Implement `pcap_to_windowed_csv.py` and verify counts.
* [ ] Implement entropy CPU and plot.
* [ ] Train RF model and save `rf_model.joblib`.
* [ ] Implement OpenCL entropy kernel; validate vs CPU.
* [ ] Implement host program to run kernel, do batch ML inference, and emit alerts.
* [ ] Implement `rtbh_controller.py` + `pcap_filter.py` and test blocking on small trace.
* [ ] Add logging & experiment automation scripts.
* [ ] Run final experiments and prepare report & slides.

---

If you want, I can now immediately:

* produce the **OpenCL entropy kernel** file (`entropy.cl`) and a **host C template** (`host.c`) prefilled with buffer setup, event profiling, and example enqueue/read calls, **or**
* create the **pcap filtering blocker** Python script (scapy-based) to simulate RTBH, **or**
* generate the **train_ml.py** template for RandomForest including feature extraction from window CSVs.


Optional (recommended for demo): a lightweight Web UI (Flask + simple JS) showing:

Live GPU kernel times and throughput

Alert timeline (timestamps + IPs)

Toggle RTBH on/off

Current blackhole list
This is handy for demo/viva but not necessary for grading. If you want it, I can generate a minimal Flask dashboard scaffold.