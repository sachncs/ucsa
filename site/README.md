# UCSA · Site

Premium product site for the
[UCSA — Unified Cognitive State Architecture](https://github.com/sachncs/ucsa)
research project.

The site is a Vite + React + TypeScript single-page application, deployed to
GitHub Pages from `site/dist`. All content is hand-curated from the
repository's README, docs, and paper draft — no docs are rendered as-is.

## Develop

```bash
cd site
npm install
npm run dev          # http:// 5173
npm run typecheck    # strict TS
npm run build        # production build into dist
npm run preview      # preview dist locally
```

## Structure

```
site/
├── public/            # static assets (favicon, og, SVGs)
├── src/
│   ├── components/    # Nav, Footer, Brand, icons
│   ├── sections/      # Hero, Concept, Architecture, Banks, …
│   ├── lib/           # content data, hooks, style helpers
│   ├── styles/        # global.css — design tokens + components
│   ├── App.tsx        # page composition
│   └── main.tsx       # entry
├── index.html
├── vite.config.ts
├── tsconfig.json
└── package.json
```

## Deploy

GitHub Actions builds and publishes `site/dist` to GitHub Pages on every push
to `master` (`.github/workflows/pages.yml`). The site lives at
<https://sachncs.github.io/ucsa/>.

## Conventions

- Single accent (violet/indigo); light/dark balance with a dark hero.
- Tokens, primitives, components, sections — strict layering in `global.css`.
- Inter (UI), Newsreader italic (accent words), JetBrains Mono (code/eyebrows).
- Reduced-motion respected; smooth scroll only with motion enabled.