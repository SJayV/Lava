import { describe, it, expect } from 'vitest';
import { computeParticleMass } from '../../rendering/densityField.js';
import { makePairs, computeCompensatedAnchorRadius } from '../../simulation/dripState.js';

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
