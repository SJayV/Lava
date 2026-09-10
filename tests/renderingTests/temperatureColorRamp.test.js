import { describe, it, expect } from 'vitest';
import { getTemperatureColorRampShaderChunk } from '../../rendering/temperatureColorRamp.js';

describe('getTemperatureColorRampShaderChunk', () => {
  it('exposes computeTemperatureColor as a WGSL function', () => {
    expect(getTemperatureColorRampShaderChunk()).toContain('fn computeTemperatureColor(');
  });
});
