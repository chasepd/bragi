# Bragi Frontend

## Size Budgets

Production builds run a size check after Vite emits static assets and gzip
sidecars:

```bash
npm run build --prefix frontend
```

To run only the budget check against the current `bragi_web/static` output:

```bash
npm run size-budget --prefix frontend
```

Budgets live in `frontend/size-budget.json`. Generated JS and CSS are measured
as raw bytes, gzip level 9, and Brotli quality 11. Public PWA icons are measured
as raw PNG bytes and checked for expected dimensions.

When a reviewed change intentionally increases initial-load size, update
`frontend/size-budget.json` in the same commit and explain the reason in the PR.

## PWA Icons

The served PWA icons are generated from `frontend/assets/app-icon-source.png`.
After replacing the source artwork, regenerate the checked-in public icons:

```bash
npm run optimize-icons --prefix frontend
```

Inspect the generated `frontend/public/*.png` files at native and small launcher
sizes before committing.
