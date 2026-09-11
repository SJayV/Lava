import { describe, it, expect } from 'vitest';
import { getPhaseChunk } from '../../shaderChunks/phaseChunk.js';

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

describe('getPhaseChunk — structure', () => {
  const chunk = getPhaseChunk();

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

  describe('exit predicates and activation are separate, and mu is only ever set from inside a shouldExit branch', () => {
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