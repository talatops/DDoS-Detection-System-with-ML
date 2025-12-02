Project Title: High-Rate Network Traffic Analyzer for Early DDoS Detection and
Mitigation
Group Size: Three students max
Submission Format: Report + Code
What to deliver
Each group must submit:
1. Source Code (well-documented and executable).
2. Experiment Scripts (for replay, testing, and metrics collection).
3. Performance Report (10–12 pages, including all analysis).
4. Presentation Slides (for viva/demo).
Project Overview
Distributed Denial of Service (DDoS) attacks pose one of the most significant challenges

to network security. Detecting and mitigating these attacks in real time requires high-
performance computation and efficient parallel/distributed system design.

In this project, your team will design and implement a high-rate network traffic
analyzer capable of detecting and blocking DDoS attacks using multiple detection
algorithms implemented on a parallel platform.
You will analyze detection performance, blocking efficiency, and scalability.
Each group must select and implement ONE of the following scenarios:
 Scenario A: GPU-based implementation using OPENCL
 Scenario B: Cluster-based implementation using MPI

Project Objectives
1. Develop a high-performance DDoS detection system using parallel/distributed
programming techniques.
2. Implement at least two DDoS detection algorithms (from literature) on network
flow or packet data.
3. Implement at least 1 blocking/mitigation methods for detected DDoS traffic.
4. Evaluate detection accuracy, latency, throughput, and scalability.

5. Analyze and compare performance results across rates, algorithms, and
configurations.

3. Link to OBE and Complex Computing Problem
(CCP)
This semester project directly addresses a Complex Computing Problem (CCP) as
required by the Outcome-Based Education (OBE) framework of the department and
NCEAC accreditation guidelines.
 Nature of Complexity:
The problem involves the design and analysis of a parallel and distributed
system for real-time network security. It requires integrating concepts of parallel
programming, network traffic analysis, and cyber defense mechanisms—
demanding the use of advanced computing knowledge and specialized
techniques beyond standard computing practices.
 Justification as a CCP:
o The problem has no known deterministic solution and requires
research-based decision-making (choice of algorithms, data
partitioning, GPU kernel design, etc.).
o It involves multiple interacting components (data ingestion, detection,
blocking, and performance evaluation).
o The system must handle real-world complexity, including large-scale
datasets and high-speed network traffic.
o Students must consider performance trade-offs (accuracy vs. speed,
detection latency vs. resource usage).
o It aligns with Program Learning Outcomes (PLO 3 & PLO 4) — problem
analysis and design/development of solutions for complex computing
problems.

Students must explicitly discuss why this problem qualifies as a CCP in the
introduction section of their report, including:
 Complexity in computation and system design
 Multiple solution approaches considered
 Justification of the chosen parallel/distributed method
 Evaluation of solution performance and limitations

Scenarios(Attempt 1 of 2)
Scenario A – GPU-Based Analyzer (OPENCL Implementation)
Goal:
Implement a high-throughput DDoS detection system using OPENCL kernels to

parallelize heavy computations such as entropy calculation, feature extraction, or ML-
based inference.

Requirements:
1. Input traffic: Use CIC-DDoS2019 dataset (preferred) or CAIDA DDoS 2007.
2. Implement at least two detection algorithms, for example:
o Entropy-based detection
o CUSUM or PCA-based statistical detection
o Machine Learning (Random Forest / SVM / Lightweight NN)
3. Offload at least one major computation to GPU kernels using OPENCL.
4. Implement two blocking methods, such as:
o Remote Triggered Black Hole (RTBH)
o Access Control List (iptables) simulation
o BGP FlowSpec simulation
5. Measure and report:
o Detection latency (how early DDoS detected)
o Accuracy metrics (TPR, FPR, Precision, Recall, F1)
o Throughput (packets/sec or Gbps)
o GPU utilization and kernel performance
o Blocking efficiency (how much attack traffic was blocked vs legitimate)

Deliverables:
 Source code (OPENCL + host code)
 GPU kernel files and documentation
 Experiment logs and plots
 Final report

Scenario B – Cluster-Based Analyzer (MPI Implementation)
Goal:
Develop a distributed DDoS detection and blocking framework using MPI on a
simulated or real cluster (3–8 nodes).
Requirements:
1. Input traffic: Use CIC-DDoS2019 dataset (preferred) or CAIDA DDoS 2007.

2. Partition traffic or flow data among MPI nodes (hash-based, time-based, or flow-
based partitioning).

3. Implement at least three detection algorithms, for example:
o Entropy-based
o PCA or CUSUM statistical detection
o Machine Learning or time-series based
4. Use MPI processes for:
o Parallel analysis per node
o Alert aggregation and decision at a coordinator node
5. Implement two blocking mechanisms, such as:
o RTBH simulation
o FlowSpec rule generation
o Rate-limiting or ACL rules
6. Measure and report:
o Detection latency (local and global)
o MPI communication overhead
o Accuracy and false alarm rate
o Throughput (scaling with number of nodes)
o Blocking effectiveness and collateral damage

Deliverables:
 Source code (MPI/C/C++/Python)
 Scripts for running distributed experiments
 Cluster configuration and test setup details
 Final report and demo video

5. Recommended Datasets
1. CIC-DDoS2019 – Canadian Institute for Cybersecurity Dataset
2. CAIDA DDoS Attack 2007 – CAIDA Dataset
3. (Optional) MAWI Traffic Archive – for mixed background traffic.

6. Suggested DDoS Detection Algorithms (choose any
2)

1. Statistical

2. ML-Based
3. Deep Learning

7. Recommended Blocking / Mitigation Methods

Method

1. Remote Triggered Black Hole (RTBH)
2. BGP FlowSpec
3. Rate Limiting
4. ACL / iptables
5. SDN Controller

8. Performance Analysis Requirements(must ensure
all these )
Your report must include quantitative evaluation of the following metrics:
1. Detection Lead Time (ms) – time between attack start and first alert.
2. Accuracy Metrics – Precision, Recall, F1, False Positive Rate.
3. Throughput – packets/sec and Gbps processed.
4. Latency – average and 95th percentile packet processing time.
5. Resource Utilization – CPU/GPU usage, memory, and network.
6. Blocking Effectiveness – attack traffic dropped (%) and collateral impact (%).
7. Scalability Analysis – performance vs. increased input rate or cluster size.
Graphs and tables must be provided for all metrics.

12. Tools and Libraries
 Programming Languages: C
 Parallel Frameworks: OPENCL / MPI
 Packet Tools: tcpreplay, tshark, scapy, Zeek
 Analysis: pandas, numpy, matplotlib, Wireshark
 Blocking Simulation: iptables, tc, or custom scripts

13. Expected Learning Outcomes
By completing this project, students will:
 Understand the design of scalable and parallel DDoS detection systems.
 Gain hands-on experience in OPENCL or MPI for parallel/distributed computing.
 Learn to analyze real network datasets and measure system performance.
 Apply cybersecurity concepts (detection + response) to real-time systems.
 Produce an end-to-end system prototype simulating a real-world DDoS defense
pipeline.
 Demonstrate competence in solving Complex Computing Problems (CCPs)
as required by the OBE framework.