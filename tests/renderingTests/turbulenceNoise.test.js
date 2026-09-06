import { describe, it, expect } from 'vitest';
import {
  hashLattice3D,
  computeValueNoise3D,
  computeTurbulence,
  getTurbulenceNoiseShaderChunk,
} from '../../rendering/turbulenceNoise.js';

describe('hashLattice3D', () => {
  it('returns a value in [0, 1)', () => {
    for (let i = 0; i < 20; i += 1) {
      const value = hashLattice3D(i, i * 3 - 7, -i * 2);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  it('is deterministic', () => {
    expect(hashLattice3D(4, -2, 9)).toBe(hashLattice3D(4, -2, 9));
  });

  it('differs across distinct lattice points', () => {
    const values = new Set();
    for (let i = 0; i < 10; i += 1) {
      values.add(hashLattice3D(i, 0, 0));
    }
    expect(values.size).toBeGreaterThan(1);
  });
});

describe('computeValueNoise3D', () => {
  it('equals the raw hash exactly at integer lattice points', () => {
    expect(computeValueNoise3D(2, -3, 5)).toBeCloseTo(hashLattice3D(2, -3, 5), 10);
  });

  it('stays within [0, 1) between lattice points', () => {
    for (let i = 0; i < 20; i += 1) {
      const value = computeValueNoise3D(i * 0.37, -i * 0.19, i * 0.5);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  it('is deterministic', () => {
    expect(computeValueNoise3D(1.25, 3.75, -0.4)).toBe(computeValueNoise3D(1.25, 3.75, -0.4));
  });
});

describe('computeTurbulence', () => {
  it('reduces to the base noise field for a single octave', () => {
    const position = [1.3, -2.1, 0.6];
    const single = computeTurbulence({ position, time: 0, octaves: 1, frequency: 1, speed: 1 });
    expect(single).toBeCloseTo(computeValueNoise3D(1.3, -2.1, 0.6), 10);
  });

  it('adding octaves changes the result (adds detail)', () => {
    const position = [1.3, -2.1, 0.6];
    const one = computeTurbulence({ position, time: 0, octaves: 1, frequency: 1.8, speed: 0.4 });
    const four = computeTurbulence({ position, time: 0, octaves: 4, frequency: 1.8, speed: 0.4 });
    expect(one).not.toBeCloseTo(four, 5);
  });

  it('evolves over time', () => {
    const position = [1.3, -2.1, 0.6];
    const early = computeTurbulence({ position, time: 0, octaves: 3, frequency: 4, speed: 0.3 });
    const later = computeTurbulence({ position, time: 5, octaves: 3, frequency: 4, speed: 0.3 });
    expect(early).not.toBe(later);
  });

  it('stays within [0, 1)', () => {
    for (let i = 0; i < 10; i += 1) {
      const value = computeTurbulence({ position: [i * 0.7, i * -1.3, i * 0.2], time: i * 0.5, octaves: 4, frequency: 1.8, speed: 0.4 });
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });
});

describe('getTurbulenceNoiseShaderChunk', () => {
  it('exposes computeTurbulence, computeValueNoise3D and hashLattice3D as WGSL functions', () => {
    const chunk = getTurbulenceNoiseShaderChunk();
    expect(chunk).toContain('fn hashLattice3D(');
    expect(chunk).toContain('fn computeValueNoise3D(');
    expect(chunk).toContain('fn computeTurbulence(');
  });
});
