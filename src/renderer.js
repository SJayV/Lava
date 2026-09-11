import { SHADER_SOURCE } from '../shaders/raymarchShader.js';

// ───── CAMERA CONFIGURATION ─────

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

function _normalize(a) {
  const length = Math.hypot(a[0], a[1], a[2]);
  return [a[0] / length, a[1] / length, a[2] / length];
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

const UNIFORM_BUFFER_SIZE = 10 * 16;

export function makeDropRaymarcher(device, presentationFormat) {
  const uniformBuffer = device.createBuffer({
    size: UNIFORM_BUFFER_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ],
  });

  const shaderModule = device.createShaderModule({ code: SHADER_SOURCE });
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module: shaderModule, entryPoint: 'vertexMain' },
    fragment: { module: shaderModule, entryPoint: 'fragmentMain', targets: [{ format: presentationFormat }] },
    primitive: { topology: 'triangle-list' },
  });

  return { device, uniformBuffer, bindGroupLayout, pipeline };
}

// ───── UNIFORMS & RENDER PASS ─────

const DEFAULT_NOISE_SCALE = 9;
const DEFAULT_NOISE_SPEED = 0.5;
const DEFAULT_NOISE_OCTAVES = 4;

export function writeRaymarchUniforms(raymarcher, view) {
  const data = new Float32Array(UNIFORM_BUFFER_SIZE / 4);
  data.set([...view.cameraRight, 0], 0);
  data.set([...view.cameraUp, 0], 4);
  data.set([...view.cameraForward, 0], 8);
  data.set([...view.cameraEye, 0], 12);
  data.set([view.width, view.height, view.aspectRatio, view.focalLength], 16);
  data.set([...view.traceHalfExtents, view.maxRayDistance], 20);
  data.set([view.h, view.isoLevel, view.fluidDensity, view.dropCount], 24);
  data.set([view.minStep, view.maxStep, view.maxTraceSteps, 0], 28);
  data.set([...view.backgroundColor, view.gradientMagnitudeMax], 32);
  data.set([
    view.noiseScale ?? DEFAULT_NOISE_SCALE,
    view.noiseSpeed ?? DEFAULT_NOISE_SPEED,
    view.noiseOctaves ?? DEFAULT_NOISE_OCTAVES,
    view.animationTime,
  ], 36);
  raymarcher.device.queue.writeBuffer(raymarcher.uniformBuffer, 0, data);
}

export function makeRaymarchBindGroup(raymarcher, dropBuffer) {
  return raymarcher.device.createBindGroup({
    layout: raymarcher.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: raymarcher.uniformBuffer } },
      { binding: 1, resource: { buffer: dropBuffer } },
    ],
  });
}

export function renderRaymarchPass(raymarcher, commandEncoder, colorTextureView, bindGroup) {
  const pass = commandEncoder.beginRenderPass({
    colorAttachments: [{ view: colorTextureView, loadOp: 'clear', storeOp: 'store' }],
  });
  pass.setPipeline(raymarcher.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(3);
  pass.end();
}