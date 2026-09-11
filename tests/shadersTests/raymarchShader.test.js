import { describe, it, expect } from 'vitest';
import { SHADER_SOURCE } from '../../shaders/raymarchShader.js';


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
    expect(traceBody).not.toMatch(/computeDensityField\(position \+/);
  });

  it('evaluates computeDensityField at most twice per step: once for density, once for bisection', () => {
    const perStepCalls = traceBody.match(/computeDensityField\(/g) ?? [];
    expect(perStepCalls.length).toBeLessThanOrEqual(3);
  });
});

describe('traceDensityIsosurface sphere-bound acceleration', () => {
  const traceBody = _extractFunctionBody(SHADER_SOURCE, 'traceDensityIsosurface');

  it('tests every body analytically before marching', () => {
    expect(traceBody).toContain('computeRaySphereEntryExit');
  });

  it('exits immediately when no body sphere is hit', () => {
    expect(traceBody).toMatch(/anyBodyHit[\s\S]*return result/);
  });

  it('marches from the sphere-bound union, not the full trace-box range', () => {
    expect(traceBody).toContain('var t = clusterNear');
    expect(traceBody).toContain('t >= clusterFar');
  });
});

describe('traceDensityIsosurface relevant-body masking', () => {
  const traceBody = _extractFunctionBody(SHADER_SOURCE, 'traceDensityIsosurface');
  const densityBody = _extractFunctionBody(SHADER_SOURCE, 'computeDensityField');

  it('builds a relevantMask bit per hit body during the sphere-bound pass', () => {
    expect(traceBody).toMatch(/relevantMask = relevantMask \| \(1u << k\)/);
  });

  it('passes the mask into computeDensityField instead of scanning all bodies unconditionally', () => {
    expect(SHADER_SOURCE).toContain('fn computeDensityField(position: vec3<f32>, relevantMask: u32)');
    expect(densityBody).toMatch(/\(relevantMask & \(1u << i\)\) == 0u/);
  });

  it('marches using the masked density field, not the unmasked one', () => {
    const marchCalls = traceBody.match(/computeDensityField\([^)]*\)/g) ?? [];
    expect(marchCalls.length).toBeGreaterThan(0);
    for (const call of marchCalls) {
      expect(call).toContain('relevantMask');
    }
  });
});

describe('fragmentMain color noise (heatValue)', () => {
  const fragmentBody = _extractFunctionBody(SHADER_SOURCE, 'fragmentMain');

  it('samples the shared turbulence field at the hit point for heatValue', () => {
    expect(fragmentBody).toMatch(/let heatValue = computeTurbulence\(result\.position/);
  });

  it('reads noise params from the uniform buffer, not hardcoded constants', () => {
    expect(fragmentBody).toContain('uniforms.noiseParams');
  });

  it('delegates the heatValue-to-color ramp to the shared temperature ramp module', () => {
    expect(fragmentBody).toMatch(/computeTemperatureColor\(heatValue\)/);
  });
});

describe('applySurfacePerturbation', () => {
  const body = _extractFunctionBody(SHADER_SOURCE, 'applySurfacePerturbation');

  it('takes the already-computed density and the point, not the raw drop data', () => {
    expect(SHADER_SOURCE).toContain('fn applySurfacePerturbation(density: f32, position: vec3<f32>)');
  });

  it('keeps beta and the perturbation frequency/speed scoped to this function', () => {
    expect(body).toMatch(/const BETA:\s*f32\s*=/);
    expect(body).toMatch(/const PERTURBATION_FREQUENCY:\s*f32\s*=/);
    expect(body).toMatch(/const PERTURBATION_SPEED:\s*f32\s*=/);
  });

  it('calls the shared gradient noise primitive exactly once, never reimplementing it', () => {
    const calls = body.match(/computeGradientNoise3D\(/g) ?? [];
    expect(calls).toHaveLength(1);
  });

  it('does not use the expensive multi-octave field on this hot path', () => {
    expect(body).not.toContain('computeTurbulence(');
  });

  it('applies the perturbation additively to the input density', () => {
    expect(body).toMatch(/return density \+ BETA \* computeGradientNoise3D\(/);
  });
});

describe('computeDensityField perturbation', () => {
  const body = _extractFunctionBody(SHADER_SOURCE, 'computeDensityField');

  it('routes its result through applySurfacePerturbation instead of returning the raw sum', () => {
    expect(body).toMatch(/return applySurfacePerturbation\(total, position\);/);
  });
});

describe('computeFieldGradientNormal', () => {
  it('still uses a live central-difference gradient (only once per hit, not per step)', () => {
    const normalBody = _extractFunctionBody(SHADER_SOURCE, 'computeFieldGradientNormal');
    const calls = normalBody.match(/computeDensityField\(/g) ?? [];

    expect(calls).toHaveLength(6);
  });
});