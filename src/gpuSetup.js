import { SHADER_SOURCE as BLOOM_SHADER_SOURCE } from '../shaders/bloomShader.js';

// ───── GRAPHICS CONTEXT (device, canvas, presentation format) ─────

export async function initializeGraphicsContext(canvas) {
  if (!navigator.gpu) {
    throw new Error('gpuSetup: WebGPU is not available in this browser');
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('gpuSetup: no WebGPU adapter available (GPU/driver not supported, or WebGPU disabled in this browser)');
  }
  const device = await adapter.requestDevice();
  const canvasContext = canvas.getContext('webgpu');
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  function _configure() {
    const RESOLUTION_SCALE = 0.7;
    const pixelRatio = Math.min(window.devicePixelRatio, RESOLUTION_SCALE);
    canvas.width = Math.max(1, Math.floor(canvas.clientWidth * pixelRatio));
    canvas.height = Math.max(1, Math.floor(canvas.clientHeight * pixelRatio));
    canvasContext.configure({ device, format: presentationFormat, alphaMode: 'opaque' });
  }

  _configure();
  new ResizeObserver(_configure).observe(canvas);

  return { device, canvas, canvasContext, presentationFormat };
}

// ───── RESOURCE REGISTRY ─────

export function makeResourceRegistry(device) {
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

const BLOOM_DOWNSAMPLE = 2;
export const MAIN_TEXTURE_FORMAT = 'rgba16float';
const UNIFORM_BUFFER_SIZE = 2 * 16;

function _makeTexture(device, width, height) {
  return device.createTexture({
    size: [width, height],
    format: MAIN_TEXTURE_FORMAT,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

export function makePostProcessor(device, canvasFormat) {
  const uniformBuffer = device.createBuffer({
    size: UNIFORM_BUFFER_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });

  const singleTextureBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
    ],
  });
  const dualTextureBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
    ],
  });

  const shaderModule = device.createShaderModule({ code: BLOOM_SHADER_SOURCE });
  function _makePipeline(fragmentEntryPoint, bindGroupLayout, targetFormat) {
    return device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
      vertex: { module: shaderModule, entryPoint: 'vertexMain' },
      fragment: { module: shaderModule, entryPoint: fragmentEntryPoint, targets: [{ format: targetFormat }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  return {
    device,
    uniformBuffer,
    sampler,
    singleTextureBindGroupLayout,
    dualTextureBindGroupLayout,
    extractPipeline: _makePipeline('fragmentExtract', singleTextureBindGroupLayout, MAIN_TEXTURE_FORMAT),
    blurPipeline: _makePipeline('fragmentBlur', singleTextureBindGroupLayout, MAIN_TEXTURE_FORMAT),
    compositePipeline: _makePipeline('fragmentComposite', dualTextureBindGroupLayout, canvasFormat),
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

function _makeSingleTextureBindGroup(postProcessor, texture) {
  return postProcessor.device.createBindGroup({
    layout: postProcessor.singleTextureBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: postProcessor.uniformBuffer } },
      { binding: 1, resource: postProcessor.sampler },
      { binding: 2, resource: texture.createView() },
    ],
  });
}

function _makeCompositeBindGroup(postProcessor, mainTexture, bloomTexture) {
  return postProcessor.device.createBindGroup({
    layout: postProcessor.dualTextureBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: postProcessor.uniformBuffer } },
      { binding: 1, resource: postProcessor.sampler },
      { binding: 2, resource: mainTexture.createView() },
      { binding: 3, resource: bloomTexture.createView() },
    ],
  });
}

export function resizePostProcessorIfNeeded(postProcessor, width, height) {
  if (postProcessor.width === width && postProcessor.height === height) {
    return;
  }

  [postProcessor.mainTexture, postProcessor.extractTexture, postProcessor.blurATexture, postProcessor.blurBTexture]
    .filter(Boolean)
    .forEach((texture) => texture.destroy());

  const bloomWidth = Math.max(1, Math.floor(width / BLOOM_DOWNSAMPLE));
  const bloomHeight = Math.max(1, Math.floor(height / BLOOM_DOWNSAMPLE));

  postProcessor.width = width;
  postProcessor.height = height;
  postProcessor.mainTexture = _makeTexture(postProcessor.device, width, height);
  postProcessor.extractTexture = _makeTexture(postProcessor.device, bloomWidth, bloomHeight);
  postProcessor.blurATexture = _makeTexture(postProcessor.device, bloomWidth, bloomHeight);
  postProcessor.blurBTexture = _makeTexture(postProcessor.device, bloomWidth, bloomHeight);

  postProcessor.extractBindGroup = _makeSingleTextureBindGroup(postProcessor, postProcessor.mainTexture);
  postProcessor.blurHBindGroup = _makeSingleTextureBindGroup(postProcessor, postProcessor.extractTexture);
  postProcessor.blurVBindGroup = _makeSingleTextureBindGroup(postProcessor, postProcessor.blurATexture);
  postProcessor.compositeBindGroup = _makeCompositeBindGroup(postProcessor, postProcessor.mainTexture, postProcessor.blurBTexture);
}

export function getMainTextureView(postProcessor) {
  return postProcessor.mainTexture.createView();
}

function _writeBloomUniforms(postProcessor, { blurDirection, threshold, intensity, exposure }) {
  const data = new Float32Array(UNIFORM_BUFFER_SIZE / 4);
  data.set([...blurDirection, 0, 0], 0);
  data.set([threshold, intensity, exposure, 0], 4);
  postProcessor.device.queue.writeBuffer(postProcessor.uniformBuffer, 0, data);
}

function _drawFullscreenPass(commandEncoder, pipeline, bindGroup, targetView) {
  const pass = commandEncoder.beginRenderPass({
    colorAttachments: [{ view: targetView, loadOp: 'clear', storeOp: 'store' }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(3);
  pass.end();
}

const DEFAULT_BLOOM_THRESHOLD = 0.2;
const DEFAULT_BLOOM_INTENSITY = 1.1;
const DEFAULT_EXPOSURE = 1.5;

export function runPostProcessPass(postProcessor, commandEncoder, canvasTextureView, {
  threshold = DEFAULT_BLOOM_THRESHOLD,
  intensity = DEFAULT_BLOOM_INTENSITY,
  exposure = DEFAULT_EXPOSURE,
} = {}) {
  _writeBloomUniforms(postProcessor, { blurDirection: [0, 0], threshold, intensity, exposure });
  _drawFullscreenPass(commandEncoder, postProcessor.extractPipeline, postProcessor.extractBindGroup, postProcessor.extractTexture.createView());

  _writeBloomUniforms(postProcessor, { blurDirection: [1, 0], threshold, intensity, exposure });
  _drawFullscreenPass(commandEncoder, postProcessor.blurPipeline, postProcessor.blurHBindGroup, postProcessor.blurATexture.createView());

  _writeBloomUniforms(postProcessor, { blurDirection: [0, 1], threshold, intensity, exposure });
  _drawFullscreenPass(commandEncoder, postProcessor.blurPipeline, postProcessor.blurVBindGroup, postProcessor.blurBTexture.createView());

  _drawFullscreenPass(commandEncoder, postProcessor.compositePipeline, postProcessor.compositeBindGroup, canvasTextureView);
}