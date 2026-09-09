import { describe, it, expect } from 'vitest';
import {
  initializePairState,
  makePairStateRecord,
  packPairStateRecords,
  getDripPhaseSystemShaderChunk,
} from '../../simulation/dripPhaseSystem.js';

function _extractFunctionBody(source, functionName) {
  const start = source.indexOf(`fn ${functionName}(`);
  expect(start, `fn ${functionName} not found in shader source`).toBeGreaterThanOrEqual(0);
  let depth = 0;
  let bodyStart = -1;
  for (let i = start; i < source.length; i += 1) {
    if (source[i] === '{') {
      if (bodyStart === -1) bodyStart = i;
      depth += 1;
    } else if (source[i] === '}') {
      depth -= 1;
      if (depth === 0) {
        return source.slice(bodyStart, i + 1);
      }
    }
  }
  throw new Error(`unbalanced braces while extracting fn ${functionName}`);
}

describe('initializePairState', () => {
  it('seeds a resting pair as ATTACHED (phaseCode 0), muAttached at tNow', () => {
    const state = initializePairState({ tNow: 5 });

    expect(state.phaseCode).toBe(0);
    expect(state.muAttached).toBe(5);
  });

  it('seeds a resting pair with muGrowing/muFalling as never-triggered sentinels, not a pre-computed gap', () => {
    const state = initializePairState({ tNow: 5 });

    expect(state.muGrowing).toBeLessThan(-1e6);
    expect(state.muFalling).toBeLessThan(-1e6);
  });

  it('seeds a pre-activated pair as GROWING (phaseCode 1), muGrowing at tNow', () => {
    const state = initializePairState({ tNow: 5, startGrowing: true });

    expect(state.phaseCode).toBe(1);
    expect(state.muGrowing).toBe(5);
  });

  it('seeds a pre-activated pair with muAttached/muFalling as never-triggered sentinels', () => {
    const state = initializePairState({ tNow: 5, startGrowing: true });

    expect(state.muAttached).toBeLessThan(-1e6);
    expect(state.muFalling).toBeLessThan(-1e6);
  });
});

describe('makePairStateRecord / packPairStateRecords', () => {
  it('packs phase code and the three mu values in order', () => {
    const record = makePairStateRecord({ phaseCode: 1, muAttached: 1, muGrowing: 2, muFalling: 3 });

    expect(record).toEqual([1, 1, 2, 3]);
  });

  it('flattens multiple records into one Float32Array', () => {
    const packed = packPairStateRecords([
      { phaseCode: 0, muAttached: 0, muGrowing: 1, muFalling: -1e9 },
      { phaseCode: 1, muAttached: -1, muGrowing: 0, muFalling: -1e9 },
    ]);

    expect(packed).toBeInstanceOf(Float32Array);
    expect(packed.length).toBe(8);
    expect(packed[0]).toBe(0);
    expect(packed[4]).toBe(1);
  });
});

// the Gaussian bump / weight / scheduler math only ever runs on the GPU
// (the geometric triggers it depends on — separation, dripY — live in the
// compute shader's own state); tested structurally here rather than
// duplicated as a parallel JS implementation
describe('getDripPhaseSystemShaderChunk — structure', () => {
  const chunk = getDripPhaseSystemShaderChunk();

  it('exposes the dispatcher, weight, and blend functions', () => {
    expect(chunk).toContain('fn scheduleTick(');
    expect(chunk).toContain('fn computePhaseWeights(');
    expect(chunk).toContain('fn blendPhaseValue(');
    expect(chunk).toContain('fn computeAttachedGrowingGap(');
  });

  it('exposes one named function per phase per blended parameter, not a single triple-returning function', () => {
    expect(chunk).toContain('fn getAttachedGravity(');
    expect(chunk).toContain('fn getGrowingGravity(');
    expect(chunk).toContain('fn getFallingGravity(');
    expect(chunk).toContain('fn getAttachedDrag(');
    expect(chunk).toContain('fn getGrowingDrag(');
    expect(chunk).toContain('fn getFallingDrag(');
  });

  it('unpacks the packed PairState struct once via a shared accessor, not per call site', () => {
    expect(chunk).toContain('fn getPhaseCode(');
    expect(chunk).toContain('fn getMus(');
    const weightsBody = _extractFunctionBody(chunk, 'computePhaseWeights');
    expect(weightsBody).not.toContain('.phaseCodeAndMus.y');
    expect(weightsBody).toContain('getMus(');
  });

  it('the dispatcher routes to exactly one per-phase handler based on the phase code', () => {
    const body = _extractFunctionBody(chunk, 'scheduleTick');

    expect(body).toContain('scheduleAttached(');
    expect(body).toContain('scheduleGrowing(');
    expect(body).toContain('scheduleFalling(');
  });

  // cause (threshold reached) before effect (mu set): each phase exposes a
  // <phase>ShouldExit predicate and an activate<Phase> function, and the
  // scheduler only ever calls activate<Phase> from inside the matching
  // shouldExit branch — never pre-computing the next phase's mu in advance
  describe('exit predicates and activation are separate, and mu is only ever set from inside a shouldExit branch', () => {
    it('exposes a should-exit predicate and an activation function per transition', () => {
      expect(chunk).toContain('fn attachedShouldExit(');
      expect(chunk).toContain('fn growingShouldExit(');
      expect(chunk).toContain('fn fallingShouldExit(');
      expect(chunk).toContain('fn activateGrowing(');
      expect(chunk).toContain('fn activateFalling(');
      expect(chunk).toContain('fn activateAttached(');
    });

    it('scheduleAttached only calls activateGrowing from inside an attachedShouldExit check', () => {
      const body = _extractFunctionBody(chunk, 'scheduleAttached');
      expect(body).toMatch(/if \(attachedShouldExit\([^)]*\)\) \{\s*return activateGrowing\(/);
    });

    it('scheduleGrowing only calls activateFalling from inside a growingShouldExit check', () => {
      const body = _extractFunctionBody(chunk, 'scheduleGrowing');
      expect(body).toMatch(/if \(growingShouldExit\([^)]*\)\) \{\s*return activateFalling\(/);
    });

    it('scheduleFalling only calls activateAttached from inside a fallingShouldExit check', () => {
      const body = _extractFunctionBody(chunk, 'scheduleFalling');
      expect(body).toMatch(/if \(fallingShouldExit\([^)]*\)\) \{\s*return activateAttached\(/);
    });

    it('attachedShouldExit re-derives the hold gap from the stored entry time, nothing pre-cached', () => {
      const body = _extractFunctionBody(chunk, 'attachedShouldExit');
      expect(body).toContain('computeAttachedGrowingGap(');
      expect(body).toMatch(/getMus\(pair\)\.x/);
    });
  });

  describe('activation retires the phase it supersedes, mirroring the activation offset', () => {
    it('exposes a deactivation helper that mirrors the activation helper', () => {
      expect(chunk).toContain('fn computeBumpDeactivationMu(');
      const body = _extractFunctionBody(chunk, 'computeBumpDeactivationMu');
      expect(body).toMatch(/triggerTime\s*-\s*LEAD\s*\*\s*sigma/);
    });

    it('activateGrowing retires the attached bump it supersedes', () => {
      const body = _extractFunctionBody(chunk, 'activateGrowing');
      expect(body).toMatch(/next\.phaseCodeAndMus\.y = computeBumpDeactivationMu\(tNow, SIGMA_ATTACHED\)/);
    });

    it('activateFalling retires the growing bump it supersedes', () => {
      const body = _extractFunctionBody(chunk, 'activateFalling');
      expect(body).toMatch(/next\.phaseCodeAndMus\.z = computeBumpDeactivationMu\(tNow, SIGMA_GROWING\)/);
    });

    it('activateAttached retires the falling bump it supersedes (the respawn "plop" fix)', () => {
      const body = _extractFunctionBody(chunk, 'activateAttached');
      expect(body).toMatch(/next\.phaseCodeAndMus\.w = computeBumpDeactivationMu\(tNow, SIGMA_FALLING\)/);
    });
  });

  // GROWING and FALLING have physics-determined, unbounded durations — their
  // own bump must be re-pinned to "now" every substep while still active, or
  // it decays mid-flight even though the phase hasn't actually ended
  describe('unbounded-duration phases re-pin their own bump while active', () => {
    it('scheduleGrowing advances muGrowing unconditionally before checking its exit condition', () => {
      const body = _extractFunctionBody(chunk, 'scheduleGrowing');
      expect(body).toMatch(/next\.phaseCodeAndMus\.z = tNow;/);
    });

    it('scheduleFalling advances muFalling unconditionally before checking its exit condition', () => {
      const body = _extractFunctionBody(chunk, 'scheduleFalling');
      expect(body).toMatch(/next\.phaseCodeAndMus\.w = tNow;/);
    });
  });

  it('growing exits on the geometric separation trigger, not a timer', () => {
    const body = _extractFunctionBody(chunk, 'growingShouldExit');

    expect(body).toMatch(/separation\s*>\s*h/);
  });

  it('falling exits on the geometric respawn trigger', () => {
    const body = _extractFunctionBody(chunk, 'fallingShouldExit');

    expect(body).toMatch(/dripY\s*<\s*respawnY/);
  });

  it('weights are normalized against their own sum (always summing to 1)', () => {
    const body = _extractFunctionBody(chunk, 'computePhaseWeights');

    expect(body).toMatch(/rawAttached \+ rawGrowing \+ rawFalling \+ epsilon/);
  });

  it('falling drag is derived from the base drag, not an independent literal', () => {
    const body = _extractFunctionBody(chunk, 'getFallingDrag');

    expect(body).toMatch(/BASE_DRAG\s*\*\s*FALLING_DRAG_FACTOR/);
  });

  it('weights use a fixed sigma per phase, not one derived from the mu gap', () => {
    expect(chunk).not.toContain('fn computeCrossingSigma(');
    expect(chunk).not.toContain('SIGMA_MAX');
    const body = _extractFunctionBody(chunk, 'computePhaseWeights');
    expect(body).toContain('SIGMA_ATTACHED');
    expect(body).toContain('SIGMA_GROWING');
    expect(body).toContain('SIGMA_FALLING');
  });

  it('tuning constants (LEAD, sigmas, hold range, drag) are embedded in the chunk, not threaded in as parameters', () => {
    expect(chunk).toContain('const LEAD: f32');
    expect(chunk).toContain('const SIGMA_ATTACHED: f32');
    expect(chunk).toContain('const SIGMA_GROWING: f32');
    expect(chunk).toContain('const SIGMA_FALLING: f32');
    expect(chunk).toContain('const HOLD_MIN: f32');
    expect(chunk).toContain('const HOLD_MAX: f32');
    expect(chunk).toContain('const BASE_DRAG: f32');
    expect(chunk).not.toMatch(/fn computePhaseWeights\([^)]*lead/);
    expect(chunk).not.toMatch(/fn scheduleTick\([^)]*holdMin/);
  });
});
