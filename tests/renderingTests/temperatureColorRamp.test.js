import { describe, it, expect } from 'vitest';
import { YELLOW, ORANGE, RED, computeTemperatureColor, getTemperatureColorRampShaderChunk } from '../../rendering/temperatureColorRamp.js';

function _expectCloseColor(actual, expected) {
  expected.forEach((channel, i) => expect(actual[i]).toBeCloseTo(channel, 10));
}

describe('computeTemperatureColor', () => {
  it('is yellow at heatValue 0', () => {
    _expectCloseColor(computeTemperatureColor(0), YELLOW);
  });

  it('is orange at heatValue 0.5', () => {
    _expectCloseColor(computeTemperatureColor(0.5), ORANGE);
  });

  it('is red at heatValue 1', () => {
    _expectCloseColor(computeTemperatureColor(1), RED);
  });

  it('is monotonically closer to red as heatValue rises past the midpoint', () => {
    const mid = computeTemperatureColor(0.75);
    const high = computeTemperatureColor(0.9);
    expect(high[0]).toBeLessThanOrEqual(mid[0]);
    expect(high[2]).toBeLessThanOrEqual(mid[2]);
  });
});

describe('getTemperatureColorRampShaderChunk', () => {
  it('exposes computeTemperatureColor as a WGSL function', () => {
    expect(getTemperatureColorRampShaderChunk()).toContain('fn computeTemperatureColor(');
  });
});
