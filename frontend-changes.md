# Frontend Code Quality Tools - Changes

This document describes the code quality tools added to the frontend development workflow.

## Overview

Added essential code quality tools for automatic code formatting and linting of frontend files (HTML, CSS, JavaScript).

## New Files Created

### `frontend/package.json`
NPM configuration with dev dependencies and quality check scripts.

**Dev Dependencies:**
- `prettier` (^3.4.2) - Code formatter for HTML, CSS, JS
- `eslint` (^8.57.0) - JavaScript linter
- `stylelint` (^16.12.0) - CSS linter
- `stylelint-config-standard` (^36.0.1) - Standard CSS linting rules

**Available Scripts:**
| Script | Command | Description |
|--------|---------|-------------|
| `npm run format` | `prettier --write .` | Auto-format all files |
| `npm run format:check` | `prettier --check .` | Check formatting (CI-friendly) |
| `npm run lint:js` | `eslint *.js` | Lint JavaScript files |
| `npm run lint:css` | `stylelint *.css` | Lint CSS files |
| `npm run lint` | Run both linters | Combined JS + CSS linting |
| `npm run quality` | Format check + lint | Full quality check |

### `frontend/.prettierrc`
Prettier configuration for consistent code formatting:
- 2 spaces indentation
- Semicolons enabled
- Single quotes for JavaScript
- Trailing commas (ES5 style)
- 100 character print width

### `frontend/.prettierignore`
Files excluded from Prettier formatting:
- `node_modules/`
- `package-lock.json`

### `frontend/.eslintrc.json`
ESLint configuration for JavaScript:
- Browser environment
- ES2021 syntax
- Recommended rules
- `marked` library as global
- Warnings for unused variables and prefer-const

### `frontend/.stylelintrc.json`
Stylelint configuration for CSS:
- Extends standard config
- Relaxed rules for existing codebase patterns (class naming, vendor prefixes, color notation)

### `frontend/quality-check.sh`
Executable shell script that runs all quality checks in sequence:
1. Prettier formatting check
2. ESLint JavaScript linting
3. Stylelint CSS linting

Reports overall success/failure status.

## Files Formatted

The following existing files were formatted with Prettier to establish a consistent baseline:
- `frontend/index.html`
- `frontend/script.js`
- `frontend/style.css`

## Usage

### Initial Setup
```bash
cd frontend
npm install
```

### Run Quality Checks
```bash
# Using npm scripts
npm run quality

# Or using the shell script
./quality-check.sh
```

### Auto-Format Code
```bash
npm run format
```

### Individual Checks
```bash
npm run lint:js    # JavaScript only
npm run lint:css   # CSS only
npm run format:check  # Formatting only
```
