import { describe, it, expect } from 'vitest';
import {
  computeDensityKernel,
  computeParticleMass,
  computeDensityField,
  computeIsoLevel,
} from '../../rendering/densityField.js';

describe('computeDensityKernel (poly6, W_density)', () => {
  it('peaks at r = 0 with value 315/(64*pi*h^3)', () => {
    const h = 0.3;

    const value = computeDensityKernel(0, h);

    expect(value).toBeCloseTo(315 / (64 * Math.PI * h ** 3));
  });

  it('is exactly zero at r = h', () => {
    expect(computeDensityKernel(0.3, 0.3)).toBe(0);
  });

  it('is exactly zero beyond the smoothing radius', () => {
    expect(computeDensityKernel(0.31, 0.3)).toBe(0);
  });

  it('decreases monotonically from r = 0 to r = h', () => {
    const h = 0.3;
    const samples = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3].map((r) => computeDensityKernel(r, h));

    for (let i = 1; i < samples.length; i += 1) {
      expect(samples[i]).toBeLessThanOrEqual(samples[i - 1]);
    }
  });
});

describe('computeParticleMass', () => {
  it('matches the sphere-volume mass formula', () => {
    const radius = 0.1;
    const fluidDensity = 1000;

    const mass = computeParticleMass(radius, fluidDensity);

    expect(mass).toBeCloseTo((4 / 3) * Math.PI * radius ** 3 * fluidDensity);
  });
});

describe('computeDensityField', () => {
  const h = 0.3;
  const fluidDensity = 1000;
  const radius = 0.1;
  const mass = computeParticleMass(radius, fluidDensity);

  it('equals a single particle contribution when sampled at its own center', () => {
    const drops = [{ position: [0, 0, 0], mass }];

    const density = computeDensityField([0, 0, 0], drops, h);

    expect(density).toBeCloseTo(mass * computeDensityKernel(0, h));
  });

  it('sums contributions from multiple particles', () => {
    const drops = [
      { position: [0, 0, 0], mass },
      { position: [0.1, 0, 0], mass },
    ];

    const density = computeDensityField([0, 0, 0], drops, h);
    const expected =
      mass * computeDensityKernel(0, h) + mass * computeDensityKernel(0.1, h);

    expect(density).toBeCloseTo(expected);
  });

  it('ignores particles farther than h away', () => {
    const drops = [{ position: [10, 0, 0], mass }];

    const density = computeDensityField([0, 0, 0], drops, h);

    expect(density).toBe(0);
  });
});

describe('computeIsoLevel', () => {
  it('scales a single particle peak density by the calibration factor C', () => {
    const h = 0.3;
    const mass = computeParticleMass(0.1, 1000);
    const C = 0.35;

    const isoLevel = computeIsoLevel(mass, h, C);

    expect(isoLevel).toBeCloseTo(C * mass * computeDensityKernel(0, h));
  });
});
