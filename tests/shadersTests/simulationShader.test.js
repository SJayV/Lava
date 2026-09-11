import { describe, it, expect } from 'vitest';
import { SHADER_SOURCE } from '../../shaders/simulationShader.js';

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

describe('computeMain', () => {
  const body = _extractFunctionBody(SHADER_SOURCE, 'computeMain');

  it('reads phase weights exactly once — the only phase-aware call in the function', () => {
    const matches = body.match(/computePhaseWeights\(/g) ?? [];
    expect(matches).toHaveLength(1);
  });

  it('blends both gravity and drag individually through the generic blend helper', () => {
    expect(body).toMatch(/let gravity = blendPhaseValue\(/);
    expect(body).toMatch(/let drag = blendPhaseValue\(/);
  });


  it('keeps the anchor at a fixed base radius, not a separation-dependent one', () => {
    expect(body).toContain('next.positionAndRadius = vec4<f32>(currentDrops[i].positionAndRadius.xyz, anchorBaseRadius);');
  });
});