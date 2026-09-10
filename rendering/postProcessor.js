const BLOOM_DOWNSAMPLE = 2;
export const MAIN_TEXTURE_FORMAT = 'rgba16float';

const UNIFORM_BUFFER_SIZE = 2 * 16;

// ───── WGSL SHADER ─────

export const SHADER_SOURCE = /* wgsl */ `
struct PostProcessUniforms {
  blurDirection: vec4<f32>,
  thresholdIntensityExposure: vec4<f32>,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: PostProcessUniforms;
@group(0) @binding(1) var textureSampler: sampler;
@group(0) @binding(2) var sourceTexture: texture_2d<f32>;
@group(0) @binding(3) var bloomTexture: texture_2d<f32>;

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  var out: VertexOutput;
  let position = positions[vertexIndex];
  out.position = vec4<f32>(position, 0.0, 1.0);
  out.uv = vec2<f32>(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
  return out;
}

// ───── BRIGHT-PASS EXTRACTION (PLAN.md §1.2: B(p) = max(C(p) - threshold, 0)) ─────

@fragment
fn fragmentExtract(in: VertexOutput) -> @location(0) vec4<f32> {
  let threshold = uniforms.thresholdIntensityExposure.x;
  let color = textureSample(sourceTexture, textureSampler, in.uv).rgb;
  let bright = max(color - vec3<f32>(threshold), vec3<f32>(0.0));
  return vec4<f32>(bright, 1.0);
}

// ───── SEPARABLE GAUSSIAN BLUR (9-tap, run once per direction) ─────

@fragment
fn fragmentBlur(in: VertexOutput) -> @location(0) vec4<f32> {
  let texelSize = 1.0 / vec2<f32>(textureDimensions(sourceTexture));
  let step = uniforms.blurDirection.xy * texelSize;
  var sum = vec4<f32>(0.0);
  sum = sum + textureSample(sourceTexture, textureSampler, in.uv + step * -4.0) * 0.0162;
  sum = sum + textureSample(sourceTexture, textureSampler, in.uv + step * -3.0) * 0.0540;
  sum = sum + textureSample(sourceTexture, textureSampler, in.uv + step * -2.0) * 0.1216;
  sum = sum + textureSample(sourceTexture, textureSampler, in.uv + step * -1.0) * 0.1945;
  sum = sum + textureSample(sourceTexture, textureSampler, in.uv) * 0.2270;
  sum = sum + textureSample(sourceTexture, textureSampler, in.uv + step * 1.0) * 0.1945;
  sum = sum + textureSample(sourceTexture, textureSampler, in.uv + step * 2.0) * 0.1216;
  sum = sum + textureSample(sourceTexture, textureSampler, in.uv + step * 3.0) * 0.0540;
  sum = sum + textureSample(sourceTexture, textureSampler, in.uv + step * 4.0) * 0.0162;
  return sum;
}

// ───── COMPOSITE (PLAN.md §1.2: C' = (C + I*B)*E, T = C'/(1+C'), background bypass on alpha) ─────

@fragment
fn fragmentComposite(in: VertexOutput) -> @location(0) vec4<f32> {
  let mainSample = textureSample(sourceTexture, textureSampler, in.uv);
  let bloom = textureSample(bloomTexture, textureSampler, in.uv).rgb;

  if (mainSample.a < 0.5) {
    return vec4<f32>(mainSample.rgb, 1.0);
  }

  let intensity = uniforms.thresholdIntensityExposure.y;
  let exposure = uniforms.thresholdIntensityExposure.z;
  let combined = (mainSample.rgb + intensity * bloom) * exposure;
  let tonemapped = combined / (vec3<f32>(1.0) + combined);
  return vec4<f32>(tonemapped, 1.0);
}
`;

// ───── PIPELINE SETUP ─────

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

  const shaderModule = device.createShaderModule({ code: SHADER_SOURCE });
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

// ───── UNIFORMS & RENDER PASSES ─────

function _writeUniforms(postProcessor, { blurDirection, threshold, intensity, exposure }) {
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

export function runPostProcessPass(postProcessor, commandEncoder, canvasTextureView, { threshold, intensity, exposure }) {
  _writeUniforms(postProcessor, { blurDirection: [0, 0], threshold, intensity, exposure });
  _drawFullscreenPass(commandEncoder, postProcessor.extractPipeline, postProcessor.extractBindGroup, postProcessor.extractTexture.createView());

  _writeUniforms(postProcessor, { blurDirection: [1, 0], threshold, intensity, exposure });
  _drawFullscreenPass(commandEncoder, postProcessor.blurPipeline, postProcessor.blurHBindGroup, postProcessor.blurATexture.createView());

  _writeUniforms(postProcessor, { blurDirection: [0, 1], threshold, intensity, exposure });
  _drawFullscreenPass(commandEncoder, postProcessor.blurPipeline, postProcessor.blurVBindGroup, postProcessor.blurBTexture.createView());

  _drawFullscreenPass(commandEncoder, postProcessor.compositePipeline, postProcessor.compositeBindGroup, canvasTextureView);
}
