import { POLY6_NORMALIZATION } from '../shaderChunks/shapeChunk.js';
import { SHADER_SOURCE } from '../shaders/simulationShader.js';

// ───── SIZING CONSTANTS ─────

export const N_LOCAL = 4;
export const ANCHOR_RADIUS_TO_H_RATIO = 0.3;
export const DRIP_RADIUS_TO_H_RATIO = 0.3;

// ───── PARTICLE / DENSITY-FIELD MATH ─────

const POLY6_GRADIENT_MAX_COEFFICIENT = 2.7;

export function computeDensityKernel(distance, smoothingRadius) {
  if (distance < 0 || distance >= smoothingRadius) {
    return 0;
  }
  const term = smoothingRadius ** 2 - distance ** 2;
  return (POLY6_NORMALIZATION / smoothingRadius ** 9) * term ** 3;
}

export function computeParticleMass(radius, fluidDensity) {
  return (4 / 3) * Math.PI * radius ** 3 * fluidDensity;
}

export function computeRadiusFromMass(mass, fluidDensity) {
  return Math.cbrt(mass / ((4 / 3) * Math.PI * fluidDensity));
}

export function computeIsoLevel(mass, smoothingRadius, calibrationFactor) {
  return calibrationFactor * mass * computeDensityKernel(0, smoothingRadius);
}

export function computeIsoLevelCalibrationFactorForRadiusRatio(radiusRatio) {
  return (1 - radiusRatio ** 2) ** 3;
}

export const ISO_LEVEL_C = computeIsoLevelCalibrationFactorForRadiusRatio(DRIP_RADIUS_TO_H_RATIO);

export function computeIsoConsistentRadius({ isoLevel, radiusRatio, smoothingRadius, fluidDensity }) {
  const calibrationFactor = computeIsoLevelCalibrationFactorForRadiusRatio(radiusRatio);
  const requiredMass = isoLevel / (calibrationFactor * computeDensityKernel(0, smoothingRadius));
  return computeRadiusFromMass(requiredMass, fluidDensity);
}

export function computeGradientMagnitudeBound(mass, smoothingRadius, localNeighborCount) {
  const gradientKernelMax = POLY6_GRADIENT_MAX_COEFFICIENT / smoothingRadius ** 4;
  return localNeighborCount * mass * gradientKernelMax;
}

// ───── PIPELINE SETUP ─────

const UNIFORM_BUFFER_SIZE = 2 * 16;

export function makeDripComputePass(device) {
  const uniformBuffer = device.createBuffer({
    size: UNIFORM_BUFFER_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });

  const shaderModule = device.createShaderModule({ code: SHADER_SOURCE });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: shaderModule, entryPoint: 'computeMain' },
  });

  return { device, uniformBuffer, bindGroupLayout, pipeline };
}

// ───── UNIFORMS & COMPUTE PASS ─────

export function writeDripPhysicsUniforms(computePass, view) {
  const data = new Float32Array(UNIFORM_BUFFER_SIZE / 4);
  data.set([view.h, view.tNow, view.dt, view.fluidDensity], 0);
  data.set([view.respawnY, view.baseRadius, view.dropCount, view.anchorBaseRadius], 4);
  computePass.device.queue.writeBuffer(computePass.uniformBuffer, 0, data);
}

export function makeDripComputeBindGroup(computePass, currentDropBuffer, currentPairStateBuffer, nextDropBuffer, nextPairStateBuffer) {
  return computePass.device.createBindGroup({
    layout: computePass.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: computePass.uniformBuffer } },
      { binding: 1, resource: { buffer: currentDropBuffer } },
      { binding: 2, resource: { buffer: currentPairStateBuffer } },
      { binding: 3, resource: { buffer: nextDropBuffer } },
      { binding: 4, resource: { buffer: nextPairStateBuffer } },
    ],
  });
}

export function runDripComputePass(computePass, commandEncoder, bindGroup, dropCount) {
  const pass = commandEncoder.beginComputePass();
  pass.setPipeline(computePass.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(dropCount);
  pass.end();
}