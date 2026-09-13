import { SHADER_SOURCE } from '../shaders/raymarchShader.js';
import { UNIFORM_BUFFER_SIZE } from './constants.js';
import { getDropBufferPair } from './state.js';
import { initializeGpuPass, writeUniformBuffer, initializeBindGroupsByActiveIndex, drawFullscreenPass } from './helpers.js';

// ───── CONSTANTS ─────

export const CAMERA_EYE = [0, -0.6, 4];
export const CAMERA_TARGET = [0, -0.6, 0];
export const CAMERA_UP = [0, 1, 0];
export const FOV_VERTICAL = Math.PI / 4;
export const TRACE_BOUND_MARGIN_IN_H = 1;

// ───── CAMERA PROJECTION ─────

function _subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function _cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function _length(a) {
  return Math.hypot(a[0], a[1], a[2]);
}

function _normalize(a) {
  const length = _length(a);
  return [a[0] / length, a[1] / length, a[2] / length];
}

export function computeDistance(a, b) {
  return _length(_subtract(a, b));
}

export function computeCameraBasisVectors(eyePosition, targetPosition, upDirection) {
  const forwardAxis = _normalize(_subtract(eyePosition, targetPosition));
  const rightAxis = _normalize(_cross(upDirection, forwardAxis));
  const trueUpAxis = _cross(forwardAxis, rightAxis);
  return { rightAxis, trueUpAxis, forwardAxis };
}

export function computeLineSpanWidth({ eyeDistance, fovVertical, aspectRatio }) {
  return 2 * eyeDistance * Math.tan(fovVertical / 2) * aspectRatio;
}

// ───── PIPELINE SETUP ─────

function _initializeDropRaymarcher(device, presentationFormat) {
  return initializeGpuPass(device, {
    visibility: GPUShaderStage.FRAGMENT,
    bufferTypes: ['uniform', 'read-only-storage', 'read-only-storage'],
    createPipeline: (bindGroupLayout) => {
      const shaderModule = device.createShaderModule({ code: SHADER_SOURCE });
      return device.createRenderPipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        vertex: { module: shaderModule, entryPoint: 'vertexRenderScene' },
        fragment: { module: shaderModule, entryPoint: 'fragmentRenderScene', targets: [{ format: presentationFormat }] },
        primitive: { topology: 'triangle-list' },
      });
    },
  });
}

// ───── UNIFORMS & RENDER PASS ─────

function _writeRaymarchUniforms(raymarcher, { noiseScale = 9, noiseSpeed = 0.5, noiseOctaves = 4, ...view }) {
  const data = new Float32Array(UNIFORM_BUFFER_SIZE / 4);
  data.set([...view.cameraRight, 0], 0);
  data.set([...view.cameraUp, 0], 4);
  data.set([...view.cameraForward, 0], 8);
  data.set([...view.cameraEye, 0], 12);
  data.set([view.width, view.height, view.aspectRatio, view.focalLength], 16);
  data.set([...view.traceHalfExtents, view.maxRayDistance], 20);
  data.set([view.h, view.isoLevel, view.fluidDensity, view.pairCount], 24);
  data.set([view.minStep, view.maxStep, view.maxTraceSteps, 0], 28);
  data.set([...view.backgroundColor, view.gradientMagnitudeMax], 32);
  data.set([noiseScale, noiseSpeed, noiseOctaves, view.animationTime], 36);
  writeUniformBuffer(raymarcher, data);
}

function _initializeRaymarchBindGroupsByActiveIndex(raymarcher, anchorBuffer, dropState) {
  const dropStatePair = getDropBufferPair(dropState);
  return initializeBindGroupsByActiveIndex(raymarcher, (activeIndex) => [anchorBuffer, dropStatePair[activeIndex]]);
}

// ───── PUBLIC INTERFACE ─────

export function initializeSceneRenderer(device, presentationFormat, anchorBuffer, dropState, view) {
  const raymarcher = _initializeDropRaymarcher(device, presentationFormat);
  const bindGroupsByActiveIndex = _initializeRaymarchBindGroupsByActiveIndex(raymarcher, anchorBuffer, dropState);
  return { raymarcher, bindGroupsByActiveIndex, dropState, ...view };
}

export function renderScene(sceneRenderer, commandEncoder, colorTextureView, { width, height, animationTime }) {
  _writeRaymarchUniforms(sceneRenderer.raymarcher, {
    cameraRight: sceneRenderer.cameraBasis.rightAxis,
    cameraUp: sceneRenderer.cameraBasis.trueUpAxis,
    cameraForward: sceneRenderer.cameraBasis.forwardAxis,
    cameraEye: CAMERA_EYE,
    width,
    height,
    aspectRatio: sceneRenderer.aspectRatio,
    focalLength: sceneRenderer.focalLength,
    traceHalfExtents: sceneRenderer.traceHalfExtents,
    maxRayDistance: sceneRenderer.maxRayDistance,
    h: sceneRenderer.h,
    isoLevel: sceneRenderer.isoLevel,
    fluidDensity: sceneRenderer.fluidDensity,
    pairCount: sceneRenderer.pairCount,
    minStep: sceneRenderer.minStep,
    maxStep: sceneRenderer.maxStep,
    maxTraceSteps: sceneRenderer.maxTraceSteps,
    backgroundColor: sceneRenderer.backgroundColor,
    gradientMagnitudeMax: sceneRenderer.gradientMagnitudeMax,
    animationTime,
  });
  drawFullscreenPass(commandEncoder, sceneRenderer.raymarcher.pipeline, sceneRenderer.bindGroupsByActiveIndex[sceneRenderer.dropState.activeIndex], colorTextureView);
}