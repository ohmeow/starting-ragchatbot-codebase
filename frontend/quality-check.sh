#!/bin/bash

# Frontend Quality Check Script
# Runs all code quality checks for the frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Frontend Code Quality Checks"
echo "=========================================="
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
    echo ""
fi

# Track overall status
ERRORS=0

# Format check
echo "1. Checking code formatting (Prettier)..."
if npm run format:check; then
    echo "   Formatting check passed"
else
    echo "   Formatting issues found. Run 'npm run format' to fix."
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ESLint
echo "2. Linting JavaScript (ESLint)..."
if npm run lint:js; then
    echo "   JavaScript lint passed"
else
    echo "   JavaScript lint issues found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Stylelint
echo "3. Linting CSS (Stylelint)..."
if npm run lint:css; then
    echo "   CSS lint passed"
else
    echo "   CSS lint issues found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Summary
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "All quality checks passed!"
    exit 0
else
    echo "Quality checks failed with $ERRORS error(s)"
    exit 1
fi
