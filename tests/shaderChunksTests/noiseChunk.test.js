import { describe, it, expect } from 'vitest';
import { getNoiseChunk } from '../../shaderChunks/noiseChunk.js';

describe('getNoiseChunk', () => {
  it('computeGradientNoise3D builds its corner gradients from the shared hash, not a reimplementation', () => {
    const chunk = getNoiseChunk();
    const start = chunk.indexOf('fn hashGradient3D(');
    const end = chunk.indexOf('fn computeValueNoise3D(');
    const body = chunk.slice(start, end);

    expect(body).toContain('fn hashGradient3D(');
    expect(body).toContain('fn computeGradientNoise3D(');
    expect((body.match(/hashGradient3D\(/g) ?? []).length).toBe(9);
  });
});