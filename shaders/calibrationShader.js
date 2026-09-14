import { getDensityKernelChunk, getParticleMassChunk } from '../shaderChunks/shapeChunk.js';

export const SHADER_SOURCE = `
struct CalibrationUniforms {
  hRatioFluidDensityNLocal: vec4<f32>,
}

struct CalibrationResult {
  values: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: CalibrationUniforms;
@group(0) @binding(1) var<storage, read_write> result: CalibrationResult;

${getDensityKernelChunk()}
${getParticleMassChunk()}

// ───── HELPER FUNCTIONS - CALIBRATION ─────

fn computeGradientMagnitudeMax(localNeighborCount: f32, mass: f32, h: f32) -> f32 {
  const GRADIENT_COEFFICIENT: f32 = 2.7;
  return localNeighborCount * mass * (GRADIENT_COEFFICIENT / pow(h, 4.0));
}

// ───── PUBLIC INTERFACE ─────

@compute @workgroup_size(1)
fn computeCalibrationValues() {
  let h = uniforms.hRatioFluidDensityNLocal.x;
  let radiusRatio = uniforms.hRatioFluidDensityNLocal.y;
  let fluidDensity = uniforms.hRatioFluidDensityNLocal.z;
  let nLocal = uniforms.hRatioFluidDensityNLocal.w;

  let radius = radiusRatio * h;
  let mass = computeParticleMass(radius, fluidDensity);
  let isoLevel = mass * computeDensityKernel(radius, h);
  let gradientMagnitudeMax = computeGradientMagnitudeMax(nLocal, mass, h);

  result.values = vec4<f32>(radius, isoLevel, gradientMagnitudeMax, 0.0);
}
`;