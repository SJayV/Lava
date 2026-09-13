import { SHADER_SOURCE } from '../shaders/simulationShader.js';
import { SHADER_SOURCE as CALIBRATION_SHADER_SOURCE } from '../shaders/calibrationShader.js';
import { UNIFORM_BUFFER_SIZE } from './constants.js';
import { getDropBufferPair, getPairStateBufferPair, swapDropState, swapPairState } from './state.js';
import { writeUniformBuffer, initializeGpuPass, initializeBufferBindGroup, initializeBindGroupsByActiveIndex, dispatchComputePass } from './helpers.js';

// ───── SIZING CONSTANTS ─────

export const N_LOCAL = 4;
export const ANCHOR_RADIUS_TO_H_RATIO = 0.3;
export const DRIP_RADIUS_TO_H_RATIO = 0.3;

// ───── CALIBRATION PASS ─────

const CALIBRATION_RESULT_SIZE = 16;

function _initializeCalibrationPass(device) {
  return initializeGpuPass(device, {
    visibility: GPUShaderStage.COMPUTE,
    bufferTypes: ['uniform', 'storage'],
    createPipeline: (bindGroupLayout) => {
      const shaderModule = device.createShaderModule({ code: CALIBRATION_SHADER_SOURCE });
      return device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shaderModule, entryPoint: 'computeCalibrationValues' },
      });
    },
  });
}

function _writeCalibrationUniforms(calibrationPass, { smoothingRadius, dripRadiusRatio, anchorRadiusRatio, fluidDensity, localNeighborCount }) {
  const data = new Float32Array(UNIFORM_BUFFER_SIZE / 4);
  data.set([smoothingRadius, dripRadiusRatio, anchorRadiusRatio, fluidDensity], 0);
  data.set([localNeighborCount, 0, 0, 0], 4);
  writeUniformBuffer(calibrationPass, data);
}

function _runCalibrationPass(calibrationPass, bindGroup, resultBuffer, stagingBuffer) {
  const commandEncoder = calibrationPass.device.createCommandEncoder();
  dispatchComputePass(commandEncoder, calibrationPass.pipeline, bindGroup, 1);
  commandEncoder.copyBufferToBuffer(resultBuffer, 0, stagingBuffer, 0, CALIBRATION_RESULT_SIZE);
  calibrationPass.device.queue.submit([commandEncoder.finish()]);
}

async function _readCalibrationResult(stagingBuffer) {
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const [dripRadius, anchorRadius, isoLevel, gradientMagnitudeMax] = new Float32Array(stagingBuffer.getMappedRange().slice(0));
  stagingBuffer.unmap();
  return { dripRadius, anchorRadius, isoLevel, gradientMagnitudeMax };
}

export async function computeCalibration(device, { smoothingRadius, dripRadiusRatio, anchorRadiusRatio, fluidDensity, localNeighborCount }) {
  const calibrationPass = _initializeCalibrationPass(device);
  const resultBuffer = device.createBuffer({ size: CALIBRATION_RESULT_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const stagingBuffer = device.createBuffer({ size: CALIBRATION_RESULT_SIZE, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const bindGroup = initializeBufferBindGroup(calibrationPass, [resultBuffer]);

  _writeCalibrationUniforms(calibrationPass, { smoothingRadius, dripRadiusRatio, anchorRadiusRatio, fluidDensity, localNeighborCount });
  _runCalibrationPass(calibrationPass, bindGroup, resultBuffer, stagingBuffer);

  return _readCalibrationResult(stagingBuffer);
}

// ───── PIPELINE SETUP ─────

function _initializeDripComputePass(device) {
  return initializeGpuPass(device, {
    visibility: GPUShaderStage.COMPUTE,
    bufferTypes: ['uniform', 'read-only-storage', 'read-only-storage', 'read-only-storage', 'storage', 'storage'],
    createPipeline: (bindGroupLayout) => {
      const shaderModule = device.createShaderModule({ code: SHADER_SOURCE });
      return device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shaderModule, entryPoint: 'computeSimulationStep' },
      });
    },
  });
}

// ───── UNIFORMS & COMPUTE PASS ─────

function _writeDripPhysicsUniforms(computePass, view) {
  const data = new Float32Array(UNIFORM_BUFFER_SIZE / 4);
  data.set([view.h, view.tNow, view.dt, view.fluidDensity], 0);
  data.set([view.respawnY, view.baseRadius, view.pairCount, 0], 4);
  writeUniformBuffer(computePass, data);
}

function _initializeDripBindGroupsByActiveIndex(computePass, anchorBuffer, dropState, pairState) {
  const dropStatePair = getDropBufferPair(dropState);
  const pairStatePair = getPairStateBufferPair(pairState);
  return initializeBindGroupsByActiveIndex(computePass, (activeIndex, otherIndex) => [
    anchorBuffer,
    dropStatePair[activeIndex],
    pairStatePair[activeIndex],
    dropStatePair[otherIndex],
    pairStatePair[otherIndex],
  ]);
}

// ───── PUBLIC INTERFACE ─────

export function initializeDripSimulation(device, anchorBuffer, dropState, pairState, { h, fluidDensity, respawnY, baseRadius, pairCount }) {
  const computePass = _initializeDripComputePass(device);
  const bindGroupsByActiveIndex = _initializeDripBindGroupsByActiveIndex(computePass, anchorBuffer, dropState, pairState);
  return { computePass, bindGroupsByActiveIndex, dropState, pairState, h, fluidDensity, respawnY, baseRadius, pairCount, elapsedTime: 0 };
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
  dispatchComputePass(commandEncoder, simulation.computePass.pipeline, simulation.bindGroupsByActiveIndex[simulation.dropState.activeIndex], simulation.pairCount);

  swapDropState(simulation.dropState);
  swapPairState(simulation.pairState);
}