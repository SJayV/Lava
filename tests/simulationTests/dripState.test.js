import { describe, it, expect } from 'vitest';
import { computeParticleMass } from '../../rendering/densityField.js';
import {
  makePairs,
  computeInitialPhases,
  computePhaseTransition,
  computeRespawnTransition,
  computeCompensatedAnchorRadius,
  encodePhase,
  decodePhase,
} from '../../simulation/dripState.js';

describe('encodePhase / decodePhase', () => {
  it('round-trips every phase', () => {
    for (const phase of ['ATTACHED', 'GROWING', 'FALLING']) {
      expect(decodePhase(encodePhase(phase))).toBe(phase);
    }
  });

  it('gives each phase a distinct numeric code', () => {
    const codes = new Set(['ATTACHED', 'GROWING', 'FALLING'].map(encodePhase));

    expect(codes.size).toBe(3);
  });
});

describe('makePairs', () => {
  it('maps each pair to consecutive anchor/drip drop-buffer indices', () => {
    const pairs = makePairs(3);

    expect(pairs).toEqual([
      { anchorIndex: 0, dripIndex: 1 },
      { anchorIndex: 2, dripIndex: 3 },
      { anchorIndex: 4, dripIndex: 5 },
    ]);
  });
});

describe('computeInitialPhases', () => {
  it('starts every pair ATTACHED except the given active pair index', () => {
    const phases = computeInitialPhases(6, 3);

    expect(phases).toEqual(['ATTACHED', 'ATTACHED', 'ATTACHED', 'GROWING', 'ATTACHED', 'ATTACHED']);
  });
});

describe('computePhaseTransition', () => {
  it('stays ATTACHED regardless of separation (no trigger yet)', () => {
    const phase = computePhaseTransition({ phase: 'ATTACHED', separation: 10, smoothingRadius: 1 });

    expect(phase).toBe('ATTACHED');
  });

  it('stays GROWING while separation is within h', () => {
    const phase = computePhaseTransition({ phase: 'GROWING', separation: 0.5, smoothingRadius: 1 });

    expect(phase).toBe('GROWING');
  });

  it('flips GROWING to FALLING once separation exceeds h', () => {
    const phase = computePhaseTransition({ phase: 'GROWING', separation: 1.01, smoothingRadius: 1 });

    expect(phase).toBe('FALLING');
  });

  it('leaves FALLING alone (respawn is a separate check)', () => {
    const phase = computePhaseTransition({ phase: 'FALLING', separation: 100, smoothingRadius: 1 });

    expect(phase).toBe('FALLING');
  });
});

// respawn goes straight back to GROWING, not ATTACHED — step 2 is
// deterministic (no random re-trigger yet, that's step 4), so landing in
// ATTACHED would freeze the pair there forever
describe('computeRespawnTransition', () => {
  it('resets FALLING to GROWING once the drip crosses respawnY', () => {
    const phase = computeRespawnTransition({ phase: 'FALLING', dripY: -5, respawnY: -3 });

    expect(phase).toBe('GROWING');
  });

  it('leaves FALLING alone above respawnY', () => {
    const phase = computeRespawnTransition({ phase: 'FALLING', dripY: 0, respawnY: -3 });

    expect(phase).toBe('FALLING');
  });

  it('leaves non-FALLING phases untouched regardless of dripY', () => {
    expect(computeRespawnTransition({ phase: 'GROWING', dripY: -100, respawnY: -3 })).toBe('GROWING');
    expect(computeRespawnTransition({ phase: 'ATTACHED', dripY: -100, respawnY: -3 })).toBe('ATTACHED');
  });
});

// keeps the local density at the anchor's own position constant regardless
// of where its drip currently is, so the drip popping back in on respawn
// doesn't read as a sudden volume change
describe('computeCompensatedAnchorRadius', () => {
  const fluidDensity = 1000;
  const anchorBaseRadius = 0.5;
  const dripBaseRadius = 0.2;
  const smoothingRadius = 1;

  it('equals the anchor base radius when the drip is coincident (separation = 0)', () => {
    const radius = computeCompensatedAnchorRadius({
      anchorBaseRadius,
      dripBaseRadius,
      separation: 0,
      smoothingRadius,
      fluidDensity,
    });

    expect(radius).toBeCloseTo(anchorBaseRadius);
  });

  it('equals the radius of anchor+drip combined mass once separation reaches h', () => {
    const radius = computeCompensatedAnchorRadius({
      anchorBaseRadius,
      dripBaseRadius,
      separation: smoothingRadius,
      smoothingRadius,
      fluidDensity,
    });

    const combinedMass =
      computeParticleMass(anchorBaseRadius, fluidDensity) + computeParticleMass(dripBaseRadius, fluidDensity);
    const expectedRadius = Math.cbrt(combinedMass / ((4 / 3) * Math.PI * fluidDensity));
    expect(radius).toBeCloseTo(expectedRadius);
  });

  it('grows monotonically as the drip moves farther away', () => {
    const options = { anchorBaseRadius, dripBaseRadius, smoothingRadius, fluidDensity };

    const near = computeCompensatedAnchorRadius({ ...options, separation: 0.2 });
    const mid = computeCompensatedAnchorRadius({ ...options, separation: 0.5 });
    const far = computeCompensatedAnchorRadius({ ...options, separation: 0.9 });

    expect(mid).toBeGreaterThan(near);
    expect(far).toBeGreaterThan(mid);
  });

  it('stays at the fully-compensated radius beyond h (no further growth once detached)', () => {
    const options = { anchorBaseRadius, dripBaseRadius, smoothingRadius, fluidDensity };

    const atH = computeCompensatedAnchorRadius({ ...options, separation: smoothingRadius });
    const farBeyondH = computeCompensatedAnchorRadius({ ...options, separation: smoothingRadius * 5 });

    expect(farBeyondH).toBeCloseTo(atH);
  });
});
