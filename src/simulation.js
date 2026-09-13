import { SHADER_SOURCE } from '../shaders/simulationShader.js';
import { SHADER_SOURCE as CALIBRATION_SHADER_SOURCE } from '../shaders/calibrationShader.js';
import { UNIFORM_BUFFER_SIZE } from './constants.js';
import { getDripBufferPair, getPairStateBufferPair, swapDripState, swapPairState } from './state.js';

// ───── SIZING CONSTANTS ─────

export const N_LOCAL = 4;
export const ANCHOR_RADIUS_TO_H_RATIO = 0.3;
export const DRIP_RADIUS_TO_H_RATIO = 0.3;

// ───── CALIBRATION PASS ─────

const CALIBRATION_RESULT_SIZE = 16;

export async function computeCalibration(device, { smoothingRadius, dripRadiusRatio, anchorRadiusRatio, fluidDensity, localNeighborCount }) {
  const uniformBuffer = device.createBuffer({ size: UNIFORM_BUFFER_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const resultBuffer = device.createBuffer({ size: CALIBRATION_RESULT_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const stagingBuffer = device.createBuffer({ size: CALIBRATION_RESULT_SIZE, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const shaderModule = device.createShaderModule({ code: CALIBRATION_SHADER_SOURCE });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: shaderModule, entryPoint: 'computeMain' },
  });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: resultBuffer } },
    ],
  });

  const uniformData = new Float32Array(UNIFORM_BUFFER_SIZE / 4);
  uniformData.set([smoothingRadius, dripRadiusRatio, anchorRadiusRatio, fluidDensity], 0);
  uniformData.set([localNeighborCount, 0, 0, 0], 4);
  device.queue.writeBuffer(uniformBuffer, 0, uniformData);

  const commandEncoder = device.createCommandEncoder();
  const pass = commandEncoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  commandEncoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, CALIBRATION_RESULT_SIZE);
  device.queue.submit([commandEncoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const [dripRadius, anchorRadius, isoLevel, gradientMagnitudeMax] = new Float32Array(stagingBuffer.getMappedRange().slice(0));
  stagingBuffer.unmap();

  return { dripRadius, anchorRadius, isoLevel, gradientMagnitudeMax };
}

// ───── PIPELINE SETUP ─────

function _initializeDripComputePass(device) {
  const uniformBuffer = device.createBuffer({
    size: UNIFORM_BUFFER_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
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

function _writeDripPhysicsUniforms(computePass, view) {
  const data = new Float32Array(UNIFORM_BUFFER_SIZE / 4);
  data.set([view.h, view.tNow, view.dt, view.fluidDensity], 0);
  data.set([view.respawnY, view.baseRadius, view.pairCount, 0], 4);
  computePass.device.queue.writeBuffer(computePass.uniformBuffer, 0, data);
}

function _initializeDripComputeBindGroup(computePass, anchorBuffer, currentDripBuffer, currentPairStateBuffer, nextDripBuffer, nextPairStateBuffer) {
  return computePass.device.createBindGroup({
    layout: computePass.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: computePass.uniformBuffer } },
      { binding: 1, resource: { buffer: anchorBuffer } },
      { binding: 2, resource: { buffer: currentDripBuffer } },
      { binding: 3, resource: { buffer: currentPairStateBuffer } },
      { binding: 4, resource: { buffer: nextDripBuffer } },
      { binding: 5, resource: { buffer: nextPairStateBuffer } },
    ],
  });
}

function _initializeDripBindGroupsByActiveIndex(computePass, anchorBuffer, dripState, pairState) {
  const [dripStateA, dripStateB] = getDripBufferPair(dripState);
  const [pairStateA, pairStateB] = getPairStateBufferPair(pairState);
  return [
    _initializeDripComputeBindGroup(computePass, anchorBuffer, dripStateA, pairStateA, dripStateB, pairStateB),
    _initializeDripComputeBindGroup(computePass, anchorBuffer, dripStateB, pairStateB, dripStateA, pairStateA),
  ];
}

function _runDripComputePass(computePass, commandEncoder, bindGroup, pairCount) {
  const pass = commandEncoder.beginComputePass();
  pass.setPipeline(computePass.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(pairCount);
  pass.end();
}

// ───── PUBLIC INTERFACE ─────

export function initializeDripSimulation(device, anchorBuffer, dripState, pairState, { h, fluidDensity, respawnY, baseRadius, pairCount }) {
  const computePass = _initializeDripComputePass(device);
  const bindGroupsByActiveIndex = _initializeDripBindGroupsByActiveIndex(computePass, anchorBuffer, dripState, pairState);
  return { computePass, bindGroupsByActiveIndex, dripState, pairState, h, fluidDensity, respawnY, baseRadius, pairCount, elapsedTime: 0 };
}

export function stepDripSimulation(simulation, commandEncoder, dt) {
  simulation.elapsedTime += dt;
  _writeDripPhysicsUniforms(simulation.computePass, {
    h: simulation.h,
    tNow: simulation.elapsedTime,
    dt,
    fluidDensity: simulation.fluidDensity,
    respawnY: simulation.respawnY,
    baseRadius: simulation.baseRadius,
    pairCount: simulation.pairCount,
  });

  _runDripComputePass(simulation.computePass, commandEncoder, simulation.bindGroupsByActiveIndex[simulation.dripState.activeIndex], simulation.pairCount);

  swapDripState(simulation.dripState);
  swapPairState(simulation.pairState);
}