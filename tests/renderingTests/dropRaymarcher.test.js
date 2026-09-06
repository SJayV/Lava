import { describe, it, expect } from 'vitest';
import { SHADER_SOURCE } from '../../rendering/dropRaymarcher.js';

// step-size divisor: analytic bound (§1.3), not live per-step gradient
// sampling — caused a GPU TDR reset. Asserted on the shader source itself.

function _extractFunctionBody(source, functionName) {
  const start = source.indexOf(`fn ${functionName}(`);
  expect(start, `fn ${functionName} not found in shader source`).toBeGreaterThanOrEqual(0);
  let depth = 0;
  let bodyStart = -1;
  for (let i = start; i < source.length; i += 1) {
    if (source[i] === '{') {
      if (bodyStart === -1) bodyStart = i;
      depth += 1;
    } else if (source[i] === '}') {
      depth -= 1;
      if (depth === 0) {
        return source.slice(bodyStart, i + 1);
      }
    }
  }
  throw new Error(`unbalanced braces while extracting fn ${functionName}`);
}

describe('traceDensityIsosurface step size', () => {
  const traceBody = _extractFunctionBody(SHADER_SOURCE, 'traceDensityIsosurface');

  it('uses the precomputed analytic gradient bound, not a live sample', () => {
    expect(traceBody).toContain('uniforms.backgroundColor.w');
  });

  it('does not evaluate the density field via offset finite differences per step', () => {
    // old bug: 3x computeDensityField(position + ...) per step
    expect(traceBody).not.toMatch(/computeDensityField\(position \+/);
  });

  it('evaluates computeDensityField at most twice per step: once for density, once for bisection', () => {
    const perStepCalls = traceBody.match(/computeDensityField\(/g) ?? [];
    // 1 initial + 1 per step + 1 bisection, never the 4x blowup
    expect(perStepCalls.length).toBeLessThanOrEqual(3);
  });
});

describe('computeFieldGradientNormal', () => {
  it('still uses a live central-difference gradient (only once per hit, not per step)', () => {
    const normalBody = _extractFunctionBody(SHADER_SOURCE, 'computeFieldGradientNormal');
    const calls = normalBody.match(/computeDensityField\(/g) ?? [];

    // 3 axes x 2 samples = 6, intentional (§1.3)
    expect(calls).toHaveLength(6);
  });
});
