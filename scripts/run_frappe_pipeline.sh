#!/bin/bash

################################################################################
# Frappe Dataset End-to-End Pipeline
################################################################################
#
# Usage:
#   ./scripts/run_frappe_pipeline.sh [--eval-only] [--train-only]
#
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
echo "  FRAPPE DATASET - COMPLETE EVALUATION PIPELINE"
echo "=============================================================================="
echo -e "${NC}"

if [ ! -f "configs/frappe_config.yaml" ]; then
    echo -e "${RED}✗ Config file not found: configs/frappe_config.yaml${NC}"
    exit 1
fi

################################################################################
# TRAINING
################################################################################

if [ "$RUN_TRAINING" = true ]; then
    echo -e "${GREEN}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 1: TRAINING MODELS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${NC}"
    
    echo "Running Frappe training pipeline..."
    python -m src.pipelines.frappe_pipeline
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Training completed${NC}"
    else
        echo -e "${RED}✗ Training failed${NC}"
        exit 1
    fi
    echo ""
fi

################################################################################
# EVALUATION
################################################################################

if [ "$RUN_EVALUATION" = true ]; then
    echo -e "${GREEN}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  STEP 2: EVALUATING WITH CONTEXT-AWARE METRICS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${NC}"
    
    echo "Running Frappe evaluator..."
    python -m evaluators.evaluate_frappe
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Evaluation completed${NC}"
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
echo "  ✓ FRAPPE PIPELINE COMPLETED"
echo "=============================================================================="
echo -e "${NC}"

echo "Results:"
echo "  • Training: outputs/frappe/"
echo "  • Evaluation: results/frappe/"
echo ""