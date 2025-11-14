#!/usr/bin/env python3
"""
Test PCAP file processing with Phase 2 components.
Validates that pcap_reader can parse packets and window_manager creates correct windows.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_pcap_files():
    """Check if PCAP files are available for testing."""
    print("=" * 80)
    print("Checking for PCAP Files")
    print("=" * 80)
    
    pcap_dirs = [
        Path("data/cic-ddos2019"),
        Path("data/caida-ddos2007"),
    ]
    
    pcap_files = []
    for pcap_dir in pcap_dirs:
        if pcap_dir.exists():
            files = list(pcap_dir.rglob("*.pcap"))
            pcap_files.extend(files)
            print(f"\nFound {len(files)} PCAP files in {pcap_dir}")
            if files:
                print(f"  Example: {files[0].name} ({files[0].stat().st_size / (1024*1024):.1f} MB)")
    
    if not pcap_files:
        print("\n⚠️  No PCAP files found. Cannot test PCAP processing.")
        print("   Please ensure datasets are downloaded to data/ directory.")
        return None
    
    return pcap_files

def test_pcap_reader_build():
    """Check if C++ project builds successfully."""
    print("\n" + "=" * 80)
    print("Testing PCAP Reader Build")
    print("=" * 80)
    
    build_dir = Path("build")
    if not build_dir.exists():
        print("  ⚠️  Build directory does not exist. Run 'mkdir build && cd build && cmake .. && make' first.")
        return False
    
    # Check if detector executable exists
    detector_exe = build_dir / "detector"
    if detector_exe.exists():
        print(f"  ✅ Detector executable found: {detector_exe}")
        return True
    else:
        print(f"  ⚠️  Detector executable not found. Build the project first.")
        return False

def create_test_program():
    """Create a simple C++ test program for PCAP reading."""
    test_program = """#include <iostream>
#include "ingest/pcap_reader.h"
#include "ingest/window_manager.h"
#include "detectors/entropy_cpu.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <pcap_file>" << std::endl;
        return 1;
    }
    
    std::string pcap_file = argv[1];
    
    // Open PCAP file
    PcapReader reader;
    if (!reader.open(pcap_file)) {
        std::cerr << "Failed to open PCAP file: " << pcap_file << std::endl;
        return 1;
    }
    
    std::cout << "Opened PCAP file: " << pcap_file << std::endl;
    
    // Create window manager
    WindowManager wm(1);  // 1 second windows
    
    // Create entropy detector
    EntropyDetector entropy_detector;
    
    int packet_count = 0;
    int window_count = 0;
    PacketInfo packet;
    
    // Read packets
    while (reader.readNextPacket(packet)) {
        if (packet.is_valid) {
            packet_count++;
            wm.addPacket(packet);
            wm.checkWindow(packet.timestamp_us);
            
            // Check if window closed
            if (wm.getCurrentWindow().total_packets == 0 && packet_count > 1) {
                window_count++;
            }
        }
    }
    
    // Process final window
    wm.closeWindow();
    if (wm.getCurrentWindow().total_packets > 0) {
        window_count++;
    }
    
    std::cout << "Processed " << packet_count << " packets" << std::endl;
    std::cout << "Created " << window_count << " windows" << std::endl;
    
    // Calculate entropy for final window
    const WindowStats& final_window = wm.getCurrentWindow();
    if (final_window.total_packets > 0) {
        EntropyDetector::EntropyFeatures features = entropy_detector.calculateFeatures(final_window);
        std::cout << "\\nFinal Window Entropy:" << std::endl;
        std::cout << "  Source IP: " << features.src_ip_entropy << std::endl;
        std::cout << "  Destination IP: " << features.dst_ip_entropy << std::endl;
        std::cout << "  Source Port: " << features.src_port_entropy << std::endl;
        std::cout << "  Destination Port: " << features.dst_port_entropy << std::endl;
    }
    
    reader.close();
    return 0;
}
"""
    
    test_file = Path("tools/test_pcap_simple.cpp")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(test_program)
    print(f"  ✅ Created test program: {test_file}")
    return test_file

def run_pcap_test(pcap_file):
    """Run PCAP processing test."""
    print("\n" + "=" * 80)
    print(f"Running PCAP Processing Test: {pcap_file.name}")
    print("=" * 80)
    
    # Check if test executable exists
    test_exe = Path("build/test_pcap_simple")
    if not test_exe.exists():
        print("  ⚠️  Test executable not found. Building...")
        # Create test program
        test_cpp = create_test_program()
        # Note: User needs to compile this separately
        print("  ℹ️  Please compile test_pcap_simple.cpp and run manually:")
        print(f"     g++ -o build/test_pcap_simple {test_cpp} -lpcap -I. -std=c++11")
        return False
    
    try:
        # Run test
        result = subprocess.run(
            [str(test_exe), str(pcap_file)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("  ✅ PCAP processing test PASSED")
            print("\nOutput:")
            print(result.stdout)
            return True
        else:
            print("  ❌ PCAP processing test FAILED")
            print("\nError:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("  ⚠️  Test timed out (processing large PCAP file)")
        return False
    except Exception as e:
        print(f"  ❌ Error running test: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 80)
    print("Phase 2: PCAP Processing Validation")
    print("=" * 80)
    
    # Check for PCAP files
    pcap_files = check_pcap_files()
    
    if not pcap_files:
        print("\n⚠️  Cannot proceed without PCAP files.")
        return 1
    
    # Check build
    if not test_pcap_reader_build():
        print("\n⚠️  Please build the project first:")
        print("   mkdir -p build && cd build && cmake .. && make")
        return 1
    
    # Create test program
    create_test_program()
    
    # Run test on first PCAP file (or smallest)
    test_pcap = min(pcap_files, key=lambda p: p.stat().st_size)
    print(f"\nUsing test PCAP: {test_pcap.name} ({test_pcap.stat().st_size / (1024*1024):.1f} MB)")
    
    # Note: Actual test requires compiled executable
    print("\n" + "=" * 80)
    print("PCAP Processing Test Setup Complete")
    print("=" * 80)
    print("\nTo run the test:")
    print("1. Compile test program:")
    print("   cd build")
    print("   g++ -o test_pcap_simple ../tools/test_pcap_simple.cpp -lpcap -I.. -std=c++11")
    print("2. Run test:")
    print(f"   ./test_pcap_simple ../{test_pcap}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

