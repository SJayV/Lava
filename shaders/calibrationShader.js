import { getDensityKernelChunk, getParticleMassChunk } from '../shaderChunks/shapeChunk.js';

export const SHADER_SOURCE = /* wgsl */ `
struct CalibrationUniforms {
  hRatiosFluidDensity: vec4<f32>,
  localNeighborCount: vec4<f32>,
}

struct CalibrationResult {
  values: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: CalibrationUniforms;
@group(0) @binding(1) var<storage, read_write> result: CalibrationResult;

${getDensityKernelChunk()}
${getParticleMassChunk()}

fn computeIsoLevelCalibrationFactor(radiusRatio: f32) -> f32 {
  return pow(1.0 - radiusRatio * radiusRatio, 3.0);
}

@compute @workgroup_size(1)
fn computeMain() {
  let h = uniforms.hRatiosFluidDensity.x;
  let dripRadiusRatio = uniforms.hRatiosFluidDensity.y;
  let anchorRadiusRatio = uniforms.hRatiosFluidDensity.z;
  let fluidDensity = uniforms.hRatiosFluidDensity.w;
  let nLocal = uniforms.localNeighborCount.x;

  let peakDensity = computeDensityKernel(0.0, h);

  let dripRadius = dripRadiusRatio * h;
  let dripMass = computeParticleMass(dripRadius, fluidDensity);
  let isoLevel = computeIsoLevelCalibrationFactor(dripRadiusRatio) * dripMass * peakDensity;

  let anchorMass = isoLevel / (computeIsoLevelCalibrationFactor(anchorRadiusRatio) * peakDensity);
  let anchorRadius = computeParticleRadius(anchorMass, fluidDensity);

  const GRADIENT_COEFFICIENT: f32 = 2.7;
  let gradientMagnitudeMax = nLocal * anchorMass * (GRADIENT_COEFFICIENT / pow(h, 4.0));

  result.values = vec4<f32>(dripRadius, anchorRadius, isoLevel, gradientMagnitudeMax);
}
`;