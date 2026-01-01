# Frontend Changes

This document describes the frontend enhancements added to the project.

---

## Part 1: Code Quality Tools

Added essential code quality tools for automatic code formatting and linting of frontend files (HTML, CSS, JavaScript).

### New Files Created

#### `frontend/package.json`
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

#### `frontend/.prettierrc`
Prettier configuration for consistent code formatting:
- 2 spaces indentation
- Semicolons enabled
- Single quotes for JavaScript
- Trailing commas (ES5 style)
- 100 character print width

#### `frontend/.prettierignore`
Files excluded from Prettier formatting:
- `node_modules/`
- `package-lock.json`

#### `frontend/.eslintrc.json`
ESLint configuration for JavaScript:
- Browser environment
- ES2021 syntax
- Recommended rules
- `marked` library as global
- Warnings for unused variables and prefer-const

#### `frontend/.stylelintrc.json`
Stylelint configuration for CSS:
- Extends standard config
- Relaxed rules for existing codebase patterns (class naming, vendor prefixes, color notation)

#### `frontend/quality-check.sh`
Executable shell script that runs all quality checks in sequence:
1. Prettier formatting check
2. ESLint JavaScript linting
3. Stylelint CSS linting

Reports overall success/failure status.

### Usage

#### Initial Setup
```bash
cd frontend
npm install
```

#### Run Quality Checks
```bash
# Using npm scripts
npm run quality

# Or using the shell script
./quality-check.sh
```

#### Auto-Format Code
```bash
npm run format
```

#### Individual Checks
```bash
npm run lint:js    # JavaScript only
npm run lint:css   # CSS only
npm run format:check  # Formatting only
```

---

## Part 2: Dark/Light Theme Toggle

Added a theme toggle button that allows users to switch between dark and light themes with smooth transitions and persistent preference storage.

### Files Modified

#### 1. `frontend/index.html`
- Added a theme toggle button positioned in the top-right corner of the page
- Button includes both sun and moon SVG icons for visual feedback
- Includes accessibility attributes (`aria-label`, `title`) for screen readers and keyboard navigation

#### 2. `frontend/style.css`

**CSS Variables Added**
Extended the `:root` CSS variables to include new theme-specific properties:
- `--code-bg`: Background color for code blocks
- `--scrollbar-thumb` / `--scrollbar-thumb-hover`: Scrollbar colors
- `--source-chip-bg`: Background for source citation chips
- `--source-link-*`: Colors for source links (background, border, text, hover states)

**Light Theme Variables**
Added `[data-theme="light"]` selector with complete light theme color palette:
- Light gray backgrounds (`#f9fafb`, `#ffffff`)
- Dark text for contrast (`#1f2937`)
- Adjusted primary colors for better visibility on light backgrounds
- Proper border and surface colors for visual hierarchy

**Theme Transition Animation**
Added smooth transition effects for theme switching:
```css
body, body *, body *::before, body *::after {
    transition: background-color 0.3s ease,
                border-color 0.3s ease,
                color 0.3s ease,
                box-shadow 0.3s ease;
}
```

**Theme Toggle Button Styles**
- Fixed position in top-right corner
- Circular button with border and shadow
- Hover and focus states for interactivity
- Icon visibility toggle based on current theme
- Responsive adjustments for mobile devices

#### 3. `frontend/script.js`

**New DOM Element Reference**
Added `themeToggle` to the list of DOM element references.

**New Functions Added**

**`initializeTheme()`**
- Called on page load
- Checks localStorage for saved theme preference
- Defaults to dark theme if no preference saved
- Applies the saved/default theme

**`toggleTheme()`**
- Event handler for the toggle button click
- Determines current theme and switches to the opposite
- Calls `setTheme()` to apply the change

**`setTheme(theme)`**
- Applies the specified theme by setting/removing `data-theme` attribute on body
- Persists the choice to localStorage

### Features

1. **Icon-based Toggle**: Sun icon in dark mode, moon icon in light mode
2. **Smooth Transitions**: All color changes animate over 0.3 seconds
3. **Persistent Preference**: Theme choice saved to localStorage
4. **Accessible**: Keyboard navigable with proper ARIA attributes
5. **Responsive**: Adjusts size on mobile devices
6. **Non-intrusive**: Fixed position doesn't interfere with content

### Usage

Click the theme toggle button in the top-right corner to switch themes. The preference is automatically saved and restored on subsequent visits.
