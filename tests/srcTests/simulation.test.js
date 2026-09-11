import { describe, it, expect } from 'vitest';
import {
  computeDensityKernel,
  computeParticleMass,
  computeRadiusFromMass,
  computeIsoLevel,
  computeGradientMagnitudeBound,
  computeIsoLevelCalibrationFactorForRadiusRatio,
  computeIsoConsistentRadius,
} from '../../src/simulation.js';

function _computeIsosurfaceRadiusRatio(calibrationFactor) {
  return Math.sqrt(1 - calibrationFactor ** (1 / 3));
}

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

describe('computeIsoLevel', () => {
  it('scales a single particle peak density by the calibration factor C', () => {
    const h = 0.3;
    const mass = computeParticleMass(0.1, 1000);
    const C = 0.35;

    const isoLevel = computeIsoLevel(mass, h, C);

    expect(isoLevel).toBeCloseTo(C * mass * computeDensityKernel(0, h));
  });

  it('crosses isoLevel at the same radius regardless of particle mass', () => {
    const h = 0.3;
    const C = 0.35;
    const rIso = _computeIsosurfaceRadiusRatio(C) * h;

    const smallMass = computeParticleMass(0.05, 1000);
    const largeMass = computeParticleMass(0.2, 1000);

    for (const mass of [smallMass, largeMass]) {
      const isoLevel = computeIsoLevel(mass, h, C);
      const densityAtRIso = mass * computeDensityKernel(rIso, h);
      expect(densityAtRIso).toBeCloseTo(isoLevel);
    }
  });

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

describe('computeIsoLevelCalibrationFactorForRadiusRatio', () => {
  it('inverts the radius-ratio-to-calibration-factor relationship', () => {
    const C = 0.6;

    const radiusRatio = _computeIsosurfaceRadiusRatio(C);
    const roundTrippedC = computeIsoLevelCalibrationFactorForRadiusRatio(radiusRatio);

    expect(roundTrippedC).toBeCloseTo(C);
  });

  it('reproduces the 2x-too-big regression: C = 0.35 renders a radius far larger than the ~0.3h drop radius', () => {
    const radiusRatio = _computeIsosurfaceRadiusRatio(0.35);

    expect(radiusRatio).toBeCloseTo(0.5434, 3);
    expect(radiusRatio / 0.3).toBeGreaterThan(1.5);
  });

  it('a higher calibration factor renders a smaller isosurface radius', () => {
    const smallC = _computeIsosurfaceRadiusRatio(0.35);
    const largeC = _computeIsosurfaceRadiusRatio(0.8);

    expect(largeC).toBeLessThan(smallC);
  });
});

describe('computeIsoConsistentRadius', () => {
  const h = 0.3;
  const fluidDensity = 1000;

  it('a body seeded with the returned radius crosses the given isoLevel exactly at the target ratio', () => {
    const dripRatio = 0.3;
    const dripMass = computeParticleMass(dripRatio * h, fluidDensity);
    const isoLevel = computeIsoLevel(dripMass, h, computeIsoLevelCalibrationFactorForRadiusRatio(dripRatio));

    const anchorRatio = 0.6;
    const anchorRadius = computeIsoConsistentRadius({ isoLevel, radiusRatio: anchorRatio, smoothingRadius: h, fluidDensity });
    const anchorMass = computeParticleMass(anchorRadius, fluidDensity);
    const rIso = anchorRatio * h;

    expect(anchorMass * computeDensityKernel(rIso, h)).toBeCloseTo(isoLevel);
  });

  it('a larger target ratio requires more mass (and thus a larger stored radius) under the same isoLevel', () => {
    const dripMass = computeParticleMass(0.3 * h, fluidDensity);
    const isoLevel = computeIsoLevel(dripMass, h, computeIsoLevelCalibrationFactorForRadiusRatio(0.3));

    const smallRatioRadius = computeIsoConsistentRadius({ isoLevel, radiusRatio: 0.3, smoothingRadius: h, fluidDensity });
    const largeRatioRadius = computeIsoConsistentRadius({ isoLevel, radiusRatio: 0.6, smoothingRadius: h, fluidDensity });

    expect(largeRatioRadius).toBeGreaterThan(smallRatioRadius);
  });

  it('reproduces the same ratio it was itself calibrated from (round-trip)', () => {
    const ratio = 0.3;
    const mass = computeParticleMass(ratio * h, fluidDensity);
    const isoLevel = computeIsoLevel(mass, h, computeIsoLevelCalibrationFactorForRadiusRatio(ratio));

    const radius = computeIsoConsistentRadius({ isoLevel, radiusRatio: ratio, smoothingRadius: h, fluidDensity });

    expect(radius).toBeCloseTo(ratio * h);
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
});