#!/bin/bash

# Batch generate walking tours for remaining schools (excluding Boston College and Cal Poly)
# Uses improved POI callout settings

cd "$(dirname "$0")"
source .venv/bin/activate

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY is not set!"
    echo "LLM features will be disabled. Set it with:"
    echo "  export OPENAI_API_KEY='sk-your-api-key-here'"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "=========================================="
echo "Generating Campus Walking Tours"
echo "Remaining Schools (with improved POI callouts)"
echo "=========================================="
echo ""

# Array of remaining schools to process
schools=(
    "Cal State San Bernardino"
    "Chico State"
    "Columbia University"
    "James Madison University"
)

total_schools=${#schools[@]}
current=1

for school in "${schools[@]}"; do
    echo ""
    echo "[$current/$total_schools] Processing: $school"
    echo "=========================================="
    
    python3 batch_run_campus_tours.py \
        --school "$school" \
        --mode consecutive \
        --llm-enabled \
        --llm-complexity medium \
        --output-dir walking_tours_improved
    
    if [ $? -eq 0 ]; then
        echo "✓ $school completed successfully"
    else
        echo "✗ $school failed (may need to retry)"
    fi
    
    current=$((current + 1))
    
    # Small delay to avoid overwhelming the Overpass API
    if [ $current -le $total_schools ]; then
        echo ""
        echo "Waiting 10 seconds before next school..."
        sleep 10
    fi
done

echo ""
echo "=========================================="
echo "Remaining schools processed!"
echo "=========================================="
echo ""
echo "Output directory: walking_tours_improved/"
echo ""
echo "All generated folders:"
ls -d walking_tours_improved/*/ 2>/dev/null | sed 's|walking_tours_improved/||; s|/||' || echo "(none yet)"
echo ""
echo "Summary:"
echo "  Boston College: ✓ (9 tours)"
echo "  Cal Poly: ✓ (8 tours)"
echo "  Cal State San Bernardino: Check output above"
echo "  Chico State: Check output above"
echo "  Columbia University: Check output above"
echo "  James Madison University: Check output above"


