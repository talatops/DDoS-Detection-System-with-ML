#!/bin/bash
# Quick status check script - see what's ready and what needs work

echo "=== DDoS Detection System - Status Check ==="
echo ""

# Check GPU
echo "1. GPU Status:"
if command -v clinfo &> /dev/null; then
    PLATFORMS=$(clinfo 2>/dev/null | grep "Number of platforms" | awk '{print $4}')
    if [ "$PLATFORMS" -gt 0 ]; then
        echo "   ✅ OpenCL working ($PLATFORMS platform(s) detected)"
        clinfo -l 2>/dev/null | grep "Device" | head -1
    else
        echo "   ❌ OpenCL not working (0 platforms)"
    fi
else
    echo "   ❌ clinfo not found"
fi

# Check datasets
echo ""
echo "2. Dataset Status:"
CSV_COUNT=$(find data -name "*.csv" 2>/dev/null | wc -l)
PCAP_COUNT=$(find data -name "*.pcap" 2>/dev/null | wc -l)
echo "   CSV files: $CSV_COUNT"
echo "   PCAP files: $PCAP_COUNT"
if [ "$CSV_COUNT" -gt 0 ] && [ "$PCAP_COUNT" -gt 0 ]; then
    echo "   ✅ Datasets available"
else
    echo "   ⚠️  Some datasets missing"
fi

# Check build
echo ""
echo "3. Build Status:"
if [ -d "build" ] && [ -f "build/Makefile" ]; then
    echo "   ✅ CMake configured"
    if [ -f "build/bin/detector" ] || [ -f "build/detector" ]; then
        echo "   ✅ Binary exists"
    else
        echo "   ⚠️  Binary not built (run 'cd build && make')"
    fi
else
    echo "   ⚠️  Not built yet (run 'mkdir build && cd build && cmake ..')"
fi

# Check ML model
echo ""
echo "4. ML Model Status:"
if [ -f "models/rf_model.joblib" ]; then
    echo "   ✅ Model exists"
    ls -lh models/rf_model.joblib
else
    echo "   ⚠️  Model not trained (run 'python3 src/ml/train_ml.py')"
fi

# Check Python packages
echo ""
echo "5. Python Packages:"
python3 -c "import sklearn, numpy, pandas, joblib" 2>/dev/null && \
    echo "   ✅ Required packages installed" || \
    echo "   ⚠️  Some packages missing (run 'pip3 install scikit-learn numpy pandas joblib')"

echo ""
echo "=== Next Steps ==="
echo ""
echo "If everything is ✅:"
echo "  1. Try building: cd build && cmake .. && make"
echo "  2. Train model: python3 src/ml/train_ml.py"
echo "  3. Test: python3 scripts/test_ml_single.py data/caida-ddos2007/CSV-01-12/01-12/Syn.csv 0"
echo ""
echo "If you see ⚠️ or ❌:"
echo "  - Fix those issues first"
echo "  - See NEXT_STEPS.md for detailed guide"
echo ""

