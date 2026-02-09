#!/bin/bash

################################################################################
# Yelp Dataset End-to-End Pipeline
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

RUN_TRAINING=true
RUN_EVALUATION=true

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
echo "  YELP DATASET - COMPLETE EVALUATION PIPELINE"
echo "=============================================================================="
echo -e "${NC}"

if [ ! -f "configs/yelp_config.yaml" ]; then
    echo -e "${RED}✗ Config file not found: configs/yelp_config.yaml${NC}"
    exit 1
fi

if [ "$RUN_TRAINING" = true ]; then
    echo -e "${GREEN}━━━━ STEP 1: TRAINING MODELS ━━━━${NC}"
    python -m src.pipelines.yelp_pipeline
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Training completed${NC}"
    else
        echo -e "${RED}✗ Training failed${NC}"
        exit 1
    fi
    echo ""
fi

if [ "$RUN_EVALUATION" = true ]; then
    echo -e "${GREEN}━━━━ STEP 2: EVALUATION ━━━━${NC}"
    python -m evaluators.evaluate_yelp
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Evaluation completed${NC}"
    else
        echo -e "${RED}✗ Evaluation failed${NC}"
        exit 1
    fi
    echo ""
fi

echo -e "${BLUE}✓ YELP PIPELINE COMPLETED${NC}"
echo "Results: results/yelp/"