# Implementation Tracker

Full spec: `PLAN.md`. Naming/structure conventions: `codingStandards.md`.

Two source directories: `rendering/` and `simulation/`, plus small shared
plumbing in `core/` and `parameters/`. Tests mirror source structure under
`tests/`: `tests/renderingTests/`, `tests/simulationTests/`,
`tests/coreTests/`, `tests/parametersTests/` (one `<module>.test.js` per
source module).

**Process per step**: write the tests for that step's behavior first
(`npm test` red), then implement against them (green), then move to the next
step. A test only gets changed if it was actually wrong — never loosened just
to make failing code pass.

## Roadmap

- [x] **Step 1 — line of metaballs, check raymarching works**
      Static line of `BALL_COUNT` drops, no physics, no noise, no shading
      beyond a flat normal-visualization. Validates the density-field
      isosurface + gradient-step raymarch loop in isolation.
      - `core/`: `parameterStore.js`, `resourceRegistry.js`, `frameLoop.js`, `graphicsContext.js`
      - `rendering/`: `cameraProjection.js`, `densityField.js`, `dropRaymarcher.js`
      - `simulation/`: `dropState.js` (seeding + GPU buffer wrapper only — no pairs/phases yet)
      - manual check: open `index.html` via a static server, confirm a line of
        blobby shapes renders (colors are normal-visualization debug output,
        not final shading)
- [x] **Step 2 — deterministic movement, check physics and tuning**
      Anchor/drip pairs (`simulation/dripState.js` or similar), the two-kernel
      force law (`W_density`, `W_cohesion`), phases (`ATTACHED`/`GROWING`/`FALLING`),
      respawn. Deterministic (no per-pair random timers yet) so behavior is
      reproducible while γ/h/μ/isoLevel get tuned. PLAN.md §2.
      - follow-up, deferred: the drip currently moves as a rigid sphere: same
        radius whether `GROWING` or `FALLING`. Add an actual elongation
        regime — the body should visually stretch into an elongated
        drop/teardrop shape while attached and pulling away, not just
        translate. Likely needs a shape parameter (e.g. anisotropic stretch
        along the anchor-drip axis, scaled by separation/h) fed into the
        density field, not just a bigger/smaller sphere radius.
- [x] **Step 3 — 3D turbulence noise, for color sampling**
      `rendering/turbulenceNoise.js` (hash + fbm), wired only into shading
      color (`heatValue`, via `rendering/temperatureColorRamp.js`) — not yet
      into surface perturbation. PLAN.md §1.4.
- [x] **Step 4 — randomness, for organic/natural movement**
      Per-pair independent random hold duration before leaving `ATTACHED`,
      derived deterministically per pair from `hashLattice3D` (PLAN.md §2.1).
      Superseded the originally-planned discrete phase FSM with a continuous
      Gaussian-bump phase-weight blend (`simulation/dripPhaseSystem.js`) for
      smooth, causally-correct transitions — see that module for the
      `<phase>ShouldExit`/`activate<Phase>` scheduler pattern. Lateral wander
      during `FALLING` not yet added.
- [ ] **Step 5 — noise, for organically perturbed surface**
      Turbulence noise folded into the density field itself
      (`Â(x,t) = A(x) + β·N(x,t)`, PLAN.md §1.3).
- [ ] **Step 6 — shading/lighting and other effects**
      `rendering/shading.js`, `rendering/temperatureColorRamp.js`,
      `rendering/postProcessor.js` (bloom pass). Replaces the Step-1 debug
      normal-visualization with real shading. PLAN.md §1.5, §1.2.

## Infrastructure

- [x] Repo scaffolded, `origin` remote wired to `https://github.com/SJayV/Lava`
- [x] Initial commit pushed
- [x] CI pipeline (GitHub Actions): run tests + lint on push/PR
      (`.github/workflows/ci.yml`)
- [x] GitHub Pages deploy — `.github/workflows/deploy.yml` copies only the
      runtime-needed files (`index.html`, `main.js`, `core/`, `rendering/`,
      `simulation/`, `parameters/`) into `_site` and deploys via
      `actions/deploy-pages`; Pages source set to "GitHub Actions" in repo
      settings
