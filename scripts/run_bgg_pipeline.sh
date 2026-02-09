#!/bin/bash

################################################################################
# BoardGameGeek (BGG) End-to-End Pipeline
################################################################################
#
# This script runs the complete evaluation pipeline for the BGG dataset:
#   1. Train models (CTR + baselines)
#   2. Generate predictions
#   3. Evaluate with context-aware metrics
#
# Usage:
#   ./scripts/run_bgg_pipeline.sh [--eval-only] [--train-only]
#
# Options:
#   --eval-only    Skip training, only run evaluation
#   --train-only   Only train models, skip evaluation
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default flags
RUN_TRAINING=true
RUN_EVALUATION=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --eval-only)
            RUN_TRAINING=false
            shift
            ;;
        --train-only)
            RUN_EVALUATION=false
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}"
echo "=============================================================================="
echo "  BGG DATASET - COMPLETE EVALUATION PIPELINE"
echo "=============================================================================="
echo -e "${NC}"

# Check if config exists
if [ ! -f "configs/bgg_config.yaml" ]; then
    echo -e "${RED}✗ Config file not found: configs/bgg_config.yaml${NC}"
    exit 1
fi

# Check if data exists
if [ ! -d "datasets/bgg" ]; then
    echo -e "${YELLOW}⚠ Warning: datasets/bgg directory not found${NC}"
    echo "  Please ensure BGG dataset is available"
fi

################################################################################
# STEP 1: TRAIN MODELS
################################################################################

if [ "$RUN_TRAINING" = true ]; then
    echo -e "${GREEN}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 1: TRAINING MODELS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${NC}"
    
    echo "Running BGG training pipeline..."
    python -m src.pipelines.bgg_pipeline
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Training completed successfully${NC}"
    else
        echo -e "${RED}✗ Training failed${NC}"
        exit 1
    fi
    
    echo ""
fi

################################################################################
# STEP 2: EVALUATE MODELS
################################################################################

if [ "$RUN_EVALUATION" = true ]; then
    echo -e "${GREEN}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 2: EVALUATING WITH CONTEXT-AWARE METRICS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${NC}"
    
    echo "Running BGG evaluator..."
    python -m evaluators.evaluate_bgg
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Evaluation completed successfully${NC}"
    else
        echo -e "${RED}✗ Evaluation failed${NC}"
        exit 1
    fi
    
    echo ""
fi

################################################################################
# COMPLETION
################################################################################

echo -e "${BLUE}"
echo "=============================================================================="
echo "  ✓ BGG PIPELINE COMPLETED"
echo "=============================================================================="
echo -e "${NC}"

echo "Results saved to:"
echo "  • Training outputs: outputs/bgg/"
echo "  • Evaluation results: results/bgg/"
echo ""
echo "Next steps:"
echo "  • View results: results/bgg/context_metrics/"
echo "  • Compare models: cat results/bgg/context_metrics/bgg_context_metrics_*.csv"
echo "  • Visualizations: results/bgg/context_metrics/*.png"
echo ""