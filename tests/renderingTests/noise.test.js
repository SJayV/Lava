import { describe, it, expect } from 'vitest';
import { getTurbulenceNoiseShaderChunk } from '../../rendering/noise.js';

describe('getTurbulenceNoiseShaderChunk', () => {
  it('exposes computeTurbulence, computeValueNoise3D, computeGradientNoise3D and hashLattice3D as WGSL functions', () => {
    const chunk = getTurbulenceNoiseShaderChunk();
    expect(chunk).toContain('fn hashLattice3D(');
    expect(chunk).toContain('fn computeValueNoise3D(');
    expect(chunk).toContain('fn computeGradientNoise3D(');
    expect(chunk).toContain('fn computeTurbulence(');
  });

  it('computeGradientNoise3D builds its corner gradients from the shared hash, not a reimplementation', () => {
    const chunk = getTurbulenceNoiseShaderChunk();
    const start = chunk.indexOf('fn hashGradient3D(');
    const end = chunk.indexOf('fn computeValueNoise3D(');
    const body = chunk.slice(start, end);

    expect(body).toContain('fn hashGradient3D(');
    expect(body).toContain('fn computeGradientNoise3D(');
    expect((body.match(/hashGradient3D\(/g) ?? []).length).toBe(9);
  });
});
