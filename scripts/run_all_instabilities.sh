#!/bin/zsh -l
# Run all instability simulations in batch mode
# Uses zsh login shell (-l) to load Vulkan SDK environment from your profile

SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data/instabilities"
OUTPUT_DIR="$PROJECT_DIR/output"
BINARY="$PROJECT_DIR/rust/target/release/proton_tracer"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Proton Radiography - Batch Processing"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo ""

# List of configurations to run
CONFIGS=(
    "zpinch"
    "sausage_weak"
    "sausage_strong"
    "kink_weak"
    "kink_strong"
    "mixed"
)

for config in "${CONFIGS[@]}"; do
    echo "----------------------------------------"
    echo "Running: $config"
    echo "----------------------------------------"

    "$BINARY" "$DATA_DIR/$config.json" --batch -o "$OUTPUT_DIR"

    echo ""
done

echo "========================================"
echo "All simulations complete!"
echo "Output files:"
ls -la "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "  (no CSV files found)"
echo "========================================"
