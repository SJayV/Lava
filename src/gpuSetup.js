import { UNIFORM_BUFFER_SIZE } from './constants.js';
import { initializeUniformBuffer, writeUniformBuffer, drawFullscreenPass } from './helpers.js';

// ───── GRAPHICS CONTEXT ─────

async function _requestGpuAdapter() {
  if (!navigator.gpu) {
    throw new Error('gpuSetup: WebGPU is not available in this browser');
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('gpuSetup: no WebGPU adapter available (GPU/driver not supported, or WebGPU disabled in this browser)');
  }
  return adapter;
}

function _computeCanvasPixelSize(canvas) {
  const pixelRatio = Math.min(window.devicePixelRatio, 0.6);
  return {
    width: Math.max(1, Math.floor(canvas.clientWidth * pixelRatio)),
    height: Math.max(1, Math.floor(canvas.clientHeight * pixelRatio)),
  };
}

function _configureCanvas(canvas, canvasContext, device, presentationFormat) {
  const { width, height } = _computeCanvasPixelSize(canvas);
  canvas.width = width;
  canvas.height = height;
  canvasContext.configure({ device, format: presentationFormat, alphaMode: 'opaque' });
}

export async function initializeGraphicsContext(canvas) {
  const adapter = await _requestGpuAdapter();
  const device = await adapter.requestDevice();
  const canvasContext = canvas.getContext('webgpu');
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  const configure = () => _configureCanvas(canvas, canvasContext, device, presentationFormat);
  configure();
  new ResizeObserver(configure).observe(canvas);

  return { device, canvas, canvasContext, presentationFormat };
}

// ───── RESOURCE REGISTRY ─────

export function initializeResourceRegistry(device) {
  return {
    device,
    buffers: new Map(),
  };
}

export function registerBuffer(registry, name, descriptor) {
  const buffer = registry.device.createBuffer(descriptor);
  registry.buffers.set(name, buffer);
  return buffer;
}

export function getBuffer(registry, name) {
  if (!registry.buffers.has(name)) {
    throw new Error(`gpuSetup: no buffer registered under "${name}"`);
  }
  return registry.buffers.get(name);
}

// ───── BLOOM SETUP ─────

export const MAIN_TEXTURE_FORMAT = 'rgba16float';

function _initializeTexture(device, width, height) {
  return device.createTexture({
    size: [width, height],
    format: MAIN_TEXTURE_FORMAT,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

function _initializeBloomSampler(device) {
  return device.createSampler({ minFilter: 'linear', magFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
}

function _initializeTextureBindGroupLayout(device, textureCount) {
  return device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ...Array.from({ length: textureCount }, (_unused, index) => ({ binding: index + 2, visibility: GPUShaderStage.FRAGMENT, texture: {} })),
    ],
  });
}

function _initializeBloomPipeline(device, shaderModule, fragmentEntryPoint, bindGroupLayout, targetFormat) {
  return device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module: shaderModule, entryPoint: 'vertexFullscreenTriangle' },
    fragment: { module: shaderModule, entryPoint: fragmentEntryPoint, targets: [{ format: targetFormat }] },
    primitive: { topology: 'triangle-list' },
  });
}

function _initializeEmptyPostProcessorTextureState() {
  return {
    width: 0,
    height: 0,
    mainTexture: null,
    extractTexture: null,
    blurATexture: null,
    blurBTexture: null,
    extractBindGroup: null,
    blurHBindGroup: null,
    blurVBindGroup: null,
    compositeBindGroup: null,
  };
}

export function initializePostProcessor(device, canvasFormat, shaderSource) {
  const uniformBuffer = initializeUniformBuffer(device);
  const sampler = _initializeBloomSampler(device);
  const singleTextureBindGroupLayout = _initializeTextureBindGroupLayout(device, 1);
  const dualTextureBindGroupLayout = _initializeTextureBindGroupLayout(device, 2);
  const shaderModule = device.createShaderModule({ code: shaderSource });

  return {
    device,
    uniformBuffer,
    sampler,
    singleTextureBindGroupLayout,
    dualTextureBindGroupLayout,
    extractPipeline: _initializeBloomPipeline(device, shaderModule, 'fragmentExtract', singleTextureBindGroupLayout, MAIN_TEXTURE_FORMAT),
    blurPipeline: _initializeBloomPipeline(device, shaderModule, 'fragmentBlur', singleTextureBindGroupLayout, MAIN_TEXTURE_FORMAT),
    compositePipeline: _initializeBloomPipeline(device, shaderModule, 'fragmentComposite', dualTextureBindGroupLayout, canvasFormat),
    ..._initializeEmptyPostProcessorTextureState(),
  };
}

function _initializeUniformAndSamplerEntries(postProcessor) {
  return [
    { binding: 0, resource: { buffer: postProcessor.uniformBuffer } },
    { binding: 1, resource: postProcessor.sampler },
  ];
}

function _initializeTextureBindGroup(postProcessor, layout, textures) {
  return postProcessor.device.createBindGroup({
    layout,
    entries: [
      ..._initializeUniformAndSamplerEntries(postProcessor),
      ...textures.map((texture, index) => ({ binding: index + 2, resource: texture.createView() })),
    ],
  });
}

function _resizeNeeded(postProcessor, width, height) {
  return postProcessor.width !== width || postProcessor.height !== height;
}

export function resizePostProcessor(postProcessor, width, height) {
  if (!_resizeNeeded(postProcessor, width, height)) {
    return;
  }

  [postProcessor.mainTexture, postProcessor.extractTexture, postProcessor.blurATexture, postProcessor.blurBTexture]
    .filter(Boolean)
    .forEach((texture) => texture.destroy());

  const BLOOM_DOWNSAMPLE = 2;
  const bloomWidth = Math.max(1, Math.floor(width / BLOOM_DOWNSAMPLE));
  const bloomHeight = Math.max(1, Math.floor(height / BLOOM_DOWNSAMPLE));

  postProcessor.width = width;
  postProcessor.height = height;
  postProcessor.mainTexture = _initializeTexture(postProcessor.device, width, height);
  postProcessor.extractTexture = _initializeTexture(postProcessor.device, bloomWidth, bloomHeight);
  postProcessor.blurATexture = _initializeTexture(postProcessor.device, bloomWidth, bloomHeight);
  postProcessor.blurBTexture = _initializeTexture(postProcessor.device, bloomWidth, bloomHeight);

  postProcessor.extractBindGroup = _initializeTextureBindGroup(postProcessor, postProcessor.singleTextureBindGroupLayout, [postProcessor.mainTexture]);
  postProcessor.blurHBindGroup = _initializeTextureBindGroup(postProcessor, postProcessor.singleTextureBindGroupLayout, [postProcessor.extractTexture]);
  postProcessor.blurVBindGroup = _initializeTextureBindGroup(postProcessor, postProcessor.singleTextureBindGroupLayout, [postProcessor.blurATexture]);
  postProcessor.compositeBindGroup = _initializeTextureBindGroup(postProcessor, postProcessor.dualTextureBindGroupLayout, [postProcessor.mainTexture, postProcessor.blurBTexture]);
}

export function getMainTextureView(postProcessor) {
  return postProcessor.mainTexture.createView();
}

function _writeBloomUniforms(postProcessor, { blurDirection, threshold, intensity, exposure }) {
  const data = new Float32Array(UNIFORM_BUFFER_SIZE / 4);
  data.set([...blurDirection, 0, 0], 0);
  data.set([threshold, intensity, exposure, 0], 4);
  writeUniformBuffer(postProcessor, data);
}

function _runBloomPass(postProcessor, commandEncoder, pipeline, bindGroup, targetView, uniforms) {
  _writeBloomUniforms(postProcessor, uniforms);
  drawFullscreenPass(commandEncoder, pipeline, bindGroup, targetView);
}

export function runPostProcessPass(postProcessor, commandEncoder, canvasTextureView, {
  threshold = 0.2,
  intensity = 1.1,
  exposure = 1.5,
} = {}) {
  _runBloomPass(postProcessor, commandEncoder, postProcessor.extractPipeline, postProcessor.extractBindGroup, postProcessor.extractTexture.createView(), { blurDirection: [0, 0], threshold, intensity, exposure });
  _runBloomPass(postProcessor, commandEncoder, postProcessor.blurPipeline, postProcessor.blurHBindGroup, postProcessor.blurATexture.createView(), { blurDirection: [1, 0], threshold, intensity, exposure });
  _runBloomPass(postProcessor, commandEncoder, postProcessor.blurPipeline, postProcessor.blurVBindGroup, postProcessor.blurBTexture.createView(), { blurDirection: [0, 1], threshold, intensity, exposure });

  drawFullscreenPass(commandEncoder, postProcessor.compositePipeline, postProcessor.compositeBindGroup, canvasTextureView);
}