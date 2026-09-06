import { describe, it, expect } from 'vitest';
import {
  computeDensityKernel,
  computeParticleMass,
  computeRadiusFromMass,
  computeDensityField,
  computeIsoLevel,
  computeGradientMagnitudeBound,
  computeIsosurfaceRadiusRatio,
  computeIsoLevelCalibrationFactorForRadiusRatio,
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
    const fluidDensity = 2000;

    const mass = computeParticleMass(radius, fluidDensity);

    expect(mass).toBeCloseTo((4 / 3) * Math.PI * radius ** 3 * fluidDensity);
  });
});

describe('computeRadiusFromMass', () => {
  it('is the inverse of computeParticleMass', () => {
    const radius = 0.15;
    const fluidDensity = 1000;

    const mass = computeParticleMass(radius, fluidDensity);
    const roundTrippedRadius = computeRadiusFromMass(mass, fluidDensity);

    expect(roundTrippedRadius).toBeCloseTo(radius);
  });

  it('grows with mass', () => {
    const fluidDensity = 1000;

    const small = computeRadiusFromMass(100, fluidDensity);
    const large = computeRadiusFromMass(800, fluidDensity);

    expect(large).toBeGreaterThan(small);
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

  // mass cancels out — rendered radius depends only on h, C
  it('crosses isoLevel at the same radius regardless of particle mass', () => {
    const h = 0.3;
    const C = 0.35;
    const rIso = computeIsosurfaceRadiusRatio(C) * h;

    const smallMass = computeParticleMass(0.05, 1000);
    const largeMass = computeParticleMass(0.2, 1000);

    for (const mass of [smallMass, largeMass]) {
      const isoLevel = computeIsoLevel(mass, h, C);
      const densityAtRIso = mass * computeDensityKernel(rIso, h);
      expect(densityAtRIso).toBeCloseTo(isoLevel);
    }
  });

  // regression: calibrating isoLevel off a heavier body's mass can leave a
  // much lighter body unable to ever cross isoLevel at all — it becomes
  // invisible once separated from the heavier body's field, even though
  // it's still physically present and moving
  it('a much lighter body cannot reach isoLevel when calibrated off a heavier one', () => {
    const h = 0.3;
    const C = 0.7;
    const heavyMass = computeParticleMass(0.3, 1000);
    const lightMass = computeParticleMass(0.05, 1000);

    const isoLevel = computeIsoLevel(heavyMass, h, C);
    const lightBodyPeakDensity = lightMass * computeDensityKernel(0, h);

    expect(lightBodyPeakDensity).toBeLessThan(isoLevel);
  });

  it('calibrating off the lighter body instead keeps it visible at its own target radius', () => {
    const h = 0.3;
    const C = 0.7;
    const lightMass = computeParticleMass(0.05, 1000);

    const isoLevel = computeIsoLevel(lightMass, h, C);
    const lightBodyPeakDensity = lightMass * computeDensityKernel(0, h);

    expect(lightBodyPeakDensity).toBeGreaterThanOrEqual(isoLevel);
  });
});

describe('computeIsosurfaceRadiusRatio / computeIsoLevelCalibrationFactorForRadiusRatio', () => {
  it('are inverses of each other', () => {
    const C = 0.6;

    const radiusRatio = computeIsosurfaceRadiusRatio(C);
    const roundTrippedC = computeIsoLevelCalibrationFactorForRadiusRatio(radiusRatio);

    expect(roundTrippedC).toBeCloseTo(C);
  });

  it('reproduces the 2x-too-big regression: C = 0.35 renders a radius far larger than the ~0.3h drop radius', () => {
    // C=0.35 -> r_iso ~0.543h, ~1.8x a 0.3h target
    const radiusRatio = computeIsosurfaceRadiusRatio(0.35);

    expect(radiusRatio).toBeCloseTo(0.5434, 3);
    expect(radiusRatio / 0.3).toBeGreaterThan(1.5);
  });

  it('a higher calibration factor renders a smaller isosurface radius', () => {
    const smallC = computeIsosurfaceRadiusRatio(0.35);
    const largeC = computeIsosurfaceRadiusRatio(0.8);

    expect(largeC).toBeLessThan(smallC);
  });
});

describe('computeGradientMagnitudeBound', () => {
  it('matches N_local * mass * (2.7 / h^4)', () => {
    const mass = computeParticleMass(0.1, 1000);
    const h = 0.3;
    const nLocal = 4;

    const bound = computeGradientMagnitudeBound(mass, h, nLocal);

    expect(bound).toBeCloseTo(nLocal * mass * (2.7 / h ** 4));
  });

  it('scales linearly with the local neighbor count', () => {
    const mass = computeParticleMass(0.1, 1000);
    const h = 0.3;

    const boundAt2 = computeGradientMagnitudeBound(mass, h, 2);
    const boundAt4 = computeGradientMagnitudeBound(mass, h, 4);

    expect(boundAt4).toBeCloseTo(boundAt2 * 2);
  });

  it('is a single closed-form expression — no drop list or dropCount input', () => {
    // no drops/dropCount — must stay O(1) per pixel
    expect(computeGradientMagnitudeBound).toHaveLength(3);
  });
});
