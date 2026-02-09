#!/bin/bash

################################################################################
# Evaluate All Datasets
################################################################################
#
# Master script to run evaluation on all three datasets:
#   - BoardGameGeek (BGG)
#   - Frappe
#   - Yelp
#
# Usage:
#   ./scripts/evaluate_all.sh [--parallel] [--sequential]
#
# Options:
#   --parallel      Run all evaluations in parallel (faster, more resource-intensive)
#   --sequential    Run evaluations one at a time (default, safer)
#
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default mode
MODE="sequential"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            MODE="parallel"
            shift
            ;;
        --sequential)
            MODE="sequential"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--parallel] [--sequential]"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║         CONTEXT-AWARE RECOMMENDATION EVALUATION - ALL DATASETS               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${BLUE}Mode: ${MODE}${NC}"
echo ""

# Function to run single dataset evaluation
run_evaluation() {
    local dataset=$1
    local script=$2
    
    echo -e "${GREEN}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  EVALUATING: ${dataset}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${NC}"
    
    if [ -f "$script" ]; then
        bash "$script" --eval-only
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ ${dataset} evaluation completed${NC}"
            return 0
        else
            echo -e "${RED}✗ ${dataset} evaluation failed${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠ Script not found: $script${NC}"
        return 1
    fi
}

# Track success/failure
declare -a RESULTS
TOTAL=0
SUCCESS=0
FAILED=0

################################################################################
# RUN EVALUATIONS
################################################################################

if [ "$MODE" = "parallel" ]; then
    echo -e "${YELLOW}Running evaluations in parallel...${NC}"
    echo ""
    
    # Run all in background
    run_evaluation "BGG" "scripts/run_bgg_pipeline.sh" &
    PID_BGG=$!
    
    run_evaluation "Frappe" "scripts/run_frappe_pipeline.sh" &
    PID_FRAPPE=$!
    
    run_evaluation "Yelp" "scripts/run_yelp_pipeline.sh" &
    PID_YELP=$!
    
    # Wait for all to complete
    echo "Waiting for all evaluations to complete..."
    
    wait $PID_BGG
    BGG_STATUS=$?
    
    wait $PID_FRAPPE
    FRAPPE_STATUS=$?
    
    wait $PID_YELP
    YELP_STATUS=$?
    
    # Record results
    RESULTS+=("BGG:$BGG_STATUS")
    RESULTS+=("Frappe:$FRAPPE_STATUS")
    RESULTS+=("Yelp:$YELP_STATUS")
    
    TOTAL=3
    [ $BGG_STATUS -eq 0 ] && ((SUCCESS++)) || ((FAILED++))
    [ $FRAPPE_STATUS -eq 0 ] && ((SUCCESS++)) || ((FAILED++))
    [ $YELP_STATUS -eq 0 ] && ((SUCCESS++)) || ((FAILED++))
    
else
    echo -e "${YELLOW}Running evaluations sequentially...${NC}"
    echo ""
    
    # BGG
    run_evaluation "BGG" "scripts/run_bgg_pipeline.sh"
    BGG_STATUS=$?
    RESULTS+=("BGG:$BGG_STATUS")
    ((TOTAL++))
    [ $BGG_STATUS -eq 0 ] && ((SUCCESS++)) || ((FAILED++))
    echo ""
    
    # Frappe
    run_evaluation "Frappe" "scripts/run_frappe_pipeline.sh"
    FRAPPE_STATUS=$?
    RESULTS+=("Frappe:$FRAPPE_STATUS")
    ((TOTAL++))
    [ $FRAPPE_STATUS -eq 0 ] && ((SUCCESS++)) || ((FAILED++))
    echo ""
    
    # Yelp
    run_evaluation "Yelp" "scripts/run_yelp_pipeline.sh"
    YELP_STATUS=$?
    RESULTS+=("Yelp:$YELP_STATUS")
    ((TOTAL++))
    [ $YELP_STATUS -eq 0 ] && ((SUCCESS++)) || ((FAILED++))
    echo ""
fi

################################################################################
# SUMMARY
################################################################################

echo -e "${CYAN}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════════════════════╗
║                          EVALUATION SUMMARY                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "Total datasets:    ${TOTAL}"
echo -e "${GREEN}Successful:        ${SUCCESS}${NC}"
echo -e "${RED}Failed:            ${FAILED}${NC}"
echo ""

echo "Detailed Results:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for result in "${RESULTS[@]}"; do
    dataset="${result%%:*}"
    status="${result##*:}"
    
    if [ "$status" -eq 0 ]; then
        echo -e "  ${GREEN}✓${NC} ${dataset}: Success"
    else
        echo -e "  ${RED}✗${NC} ${dataset}: Failed"
    fi
done

echo ""
echo "Results locations:"
echo "  • BGG:    results/bgg/context_metrics/"
echo "  • Frappe: results/frappe/context_metrics/"
echo "  • Yelp:   results/yelp/context_metrics/"
echo ""

################################################################################
# GENERATE COMPARISON REPORT (OPTIONAL)
################################################################################

if [ "$SUCCESS" -eq "$TOTAL" ]; then
    echo -e "${CYAN}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ALL EVALUATIONS COMPLETED SUCCESSFULLY!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${NC}"
    
    echo "Next steps:"
    echo "  1. Compare results across datasets"
    echo "  2. Generate visualizations"
    echo "  3. Analyze metric correlations"
    echo ""
    
    # Optional: Generate cross-dataset comparison
    if command -v python &> /dev/null; then
        echo "Would you like to generate a cross-dataset comparison report? (y/n)"
        read -r response
        
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "Generating comparison report..."
            # This would call a Python script to compare results
            # python scripts/generate_comparison_report.py
            echo -e "${YELLOW}(Comparison report generation not yet implemented)${NC}"
        fi
    fi
    
    exit 0
else
    echo -e "${RED}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  SOME EVALUATIONS FAILED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${NC}"
    
    echo "Please check the logs above for error details."
    exit 1
fi