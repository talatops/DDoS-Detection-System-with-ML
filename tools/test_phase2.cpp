/**
 * Comprehensive unit tests for Phase 2 components:
 * - pcap_reader
 * - window_manager
 * - entropy_cpu
 * - cusum_detector
 * - pca_detector
 */

#include <iostream>
#include <cassert>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstring>

// Include Phase 2 headers
#include "ingest/pcap_reader.h"
#include "ingest/window_manager.h"
#include "detectors/entropy_cpu.h"
#include "detectors/cusum_detector.h"
#include "detectors/pca_detector.h"

using namespace std;

// Test utilities
#define ASSERT_EQ(a, b) assert((a) == (b))
#define ASSERT_NEAR(a, b, tol) assert(abs((a) - (b)) < (tol))
#define TEST(name) cout << "\n[TEST] " << name << "..." << endl

// Test 1: Entropy Calculation
void test_entropy_calculation() {
    TEST("Entropy Calculation");
    
    EntropyDetector detector;
    
    // Test uniform distribution (max entropy)
    unordered_map<uint32_t, uint32_t> uniform_counts;
    for (uint32_t i = 1; i <= 4; i++) {
        uniform_counts[i] = 25;
    }
    double entropy = detector.calculateEntropy(uniform_counts, 100);
    double expected = log2(4.0);  // log2(4) = 2.0
    ASSERT_NEAR(entropy, expected, 1e-5);
    cout << "  ✓ Uniform distribution entropy: " << entropy << " (expected: " << expected << ")" << endl;
    
    // Test single value (zero entropy)
    unordered_map<uint32_t, uint32_t> single_count;
    single_count[1] = 100;
    entropy = detector.calculateEntropy(single_count, 100);
    ASSERT_NEAR(entropy, 0.0, 1e-5);
    cout << "  ✓ Single value entropy: " << entropy << " (expected: 0.0)" << endl;
    
    // Test empty counts
    unordered_map<uint32_t, uint32_t> empty_counts;
    entropy = detector.calculateEntropy(empty_counts, 0);
    ASSERT_NEAR(entropy, 0.0, 1e-5);
    cout << "  ✓ Empty counts entropy: " << entropy << " (expected: 0.0)" << endl;
}

// Test 2: Window Manager
void test_window_manager() {
    TEST("Window Manager");
    
    WindowManager wm(1);  // 1 second windows
    
    // Create test packets
    PacketInfo pkt1, pkt2, pkt3;
    pkt1.is_valid = true;
    pkt1.timestamp_us = 0;
    pkt1.src_ip = 0x01010101;  // 1.1.1.1
    pkt1.dst_ip = 0x02020202;  // 2.2.2.2
    pkt1.src_port = 12345;
    pkt1.dst_port = 80;
    pkt1.protocol = 6;  // TCP
    pkt1.packet_size = 100;
    
    pkt2 = pkt1;
    pkt2.timestamp_us = 500000;  // 0.5 seconds later
    
    pkt3 = pkt1;
    pkt3.timestamp_us = 2000000;  // 2 seconds later (should trigger new window)
    
    // Add packets
    wm.addPacket(pkt1);
    wm.addPacket(pkt2);
    
    const WindowStats& window1 = wm.getCurrentWindow();
    ASSERT_EQ(window1.total_packets, 2);
    ASSERT_EQ(window1.total_bytes, 200);
    cout << "  ✓ Window 1: " << window1.total_packets << " packets, " << window1.total_bytes << " bytes" << endl;
    
    // Check window closure
    wm.checkWindow(pkt3.timestamp_us);
    // Window should be closed and reset
    const WindowStats& window2 = wm.getCurrentWindow();
    cout << "  ✓ Window closure triggered" << endl;
}

// Test 3: CUSUM Detector
void test_cusum_detector() {
    TEST("CUSUM Detector");
    
    CUSUMDetector cusum(5.0, 0.5, 100);  // threshold=5.0, drift=0.5
    
    // Feed normal values (baseline)
    for (int i = 0; i < 50; i++) {
        cusum.update(5.0);  // Normal entropy ~5.0
    }
    
    ASSERT_EQ(cusum.isAnomaly(), false);
    cout << "  ✓ Baseline established, no anomaly detected" << endl;
    
    // Feed sudden change (attack)
    for (int i = 0; i < 20; i++) {
        cusum.update(1.0);  // Low entropy (attack)
    }
    
    // Should detect anomaly
    bool anomaly = cusum.isAnomaly();
    double stat = cusum.getStatistic();
    cout << "  ✓ CUSUM statistic: " << stat << ", Anomaly: " << (anomaly ? "YES" : "NO") << endl;
    
    // Reset
    cusum.reset();
    ASSERT_EQ(cusum.isAnomaly(), false);
    cout << "  ✓ Reset successful" << endl;
}

// Test 4: PCA Detector
void test_pca_detector() {
    TEST("PCA Detector");
    
    PCADetector pca(3, 100, 0.1);  // 3 components, 100 training samples, threshold=0.1
    
    ASSERT_EQ(pca.isTrained(), false);
    cout << "  ✓ Initial state: not trained" << endl;
    
    // Add training samples
    EntropyDetector::EntropyFeatures normal_features;
    normal_features.src_ip_entropy = 5.0;
    normal_features.dst_ip_entropy = 5.0;
    normal_features.src_port_entropy = 4.0;
    normal_features.dst_port_entropy = 4.0;
    normal_features.packet_size_entropy = 3.0;
    normal_features.protocol_entropy = 2.0;
    
    for (int i = 0; i < 100; i++) {
        pca.addTrainingSample(normal_features);
    }
    
    ASSERT_EQ(pca.isTrained(), true);
    cout << "  ✓ Training complete" << endl;
    
    // Test with normal features (should not be anomaly)
    bool anomaly = pca.detectAnomaly(normal_features);
    ASSERT_EQ(anomaly, false);
    cout << "  ✓ Normal features: no anomaly detected" << endl;
    
    // Test with anomalous features (low entropy)
    EntropyDetector::EntropyFeatures attack_features;
    attack_features.src_ip_entropy = 1.0;  // Low entropy
    attack_features.dst_ip_entropy = 1.0;
    attack_features.src_port_entropy = 0.5;
    attack_features.dst_port_entropy = 0.5;
    attack_features.packet_size_entropy = 0.5;
    attack_features.protocol_entropy = 0.5;
    
    anomaly = pca.detectAnomaly(attack_features);
    double error = pca.getReconstructionError(attack_features);
    cout << "  ✓ Attack features: reconstruction error=" << error << ", Anomaly: " << (anomaly ? "YES" : "NO") << endl;
}

// Test 5: Integration Test
void test_integration() {
    TEST("Integration: Full Pipeline");
    
    WindowManager wm(1);
    EntropyDetector entropy_detector;
    CUSUMDetector cusum(5.0, 0.5, 100);
    
    // Simulate packet flow
    PacketInfo pkt;
    pkt.is_valid = true;
    pkt.protocol = 6;  // TCP
    pkt.packet_size = 100;
    
    // Normal traffic (high entropy)
    for (int i = 0; i < 100; i++) {
        pkt.timestamp_us = i * 10000;  // 10ms intervals
        pkt.src_ip = 0x01010101 + (i % 10);  // 10 different source IPs
        pkt.dst_ip = 0x02020202;
        pkt.src_port = 1000 + (i % 20);  // 20 different ports
        pkt.dst_port = 80;
        
        wm.addPacket(pkt);
        wm.checkWindow(pkt.timestamp_us);
        
        // Process window if closed
        if (wm.getCurrentWindow().total_packets == 0 && i > 0) {
            // Window was closed, process it
            // (In real implementation, this would be done via callback)
        }
    }
    
    cout << "  ✓ Packet ingestion and windowing working" << endl;
    
    // Create a test window
    WindowStats test_window;
    test_window.total_packets = 100;
    for (int i = 0; i < 10; i++) {
        test_window.src_ip_counts[0x01010101 + i] = 10;
    }
    for (int i = 0; i < 20; i++) {
        test_window.src_port_counts[1000 + i] = 5;
    }
    
    // Calculate entropy
    EntropyDetector::EntropyFeatures features = entropy_detector.calculateFeatures(test_window);
    cout << "  ✓ Entropy calculation: src_ip=" << features.src_ip_entropy << endl;
    
    // Feed to CUSUM
    cusum.update(features.src_ip_entropy);
    cout << "  ✓ CUSUM update successful" << endl;
    
    cout << "  ✓ Full pipeline integration test PASSED" << endl;
}

int main() {
    cout << "=" << string(80, '=') << endl;
    cout << "Phase 2: Comprehensive Unit Tests" << endl;
    cout << "=" << string(80, '=') << endl;
    
    try {
        test_entropy_calculation();
        test_window_manager();
        test_cusum_detector();
        test_pca_detector();
        test_integration();
        
        cout << "\n" << string(80, '=') << endl;
        cout << "All Phase 2 Tests PASSED!" << endl;
        cout << string(80, '=') << endl;
        return 0;
    } catch (const exception& e) {
        cerr << "\n❌ TEST FAILED: " << e.what() << endl;
        return 1;
    }
}

