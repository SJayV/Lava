import { FULLSCREEN_TRIANGLE_POSITION_CHUNK } from '../src/gpuSetup.js';

export const SHADER_SOURCE = `
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

${FULLSCREEN_TRIANGLE_POSITION_CHUNK}

// ───── HELPER FUNCTIONS - VERTEX ─────

fn computeUvFromClipPosition(position: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
}

// ───── HELPER FUNCTIONS - BRIGHT-PASS EXTRACTION ─────

fn extractBrightness(color: vec3<f32>, threshold: f32) -> vec3<f32> {
  return max(color - vec3<f32>(threshold), vec3<f32>(0.0));
}

// ───── HELPER FUNCTIONS - COMPOSITE ─────

fn isBackgroundPixel(alpha: f32) -> bool {
  return alpha < 0.5;
}

fn combineBloom(color: vec3<f32>, bloom: vec3<f32>, intensity: f32, exposure: f32) -> vec3<f32> {
  return (color + intensity * bloom) * exposure;
}

fn applyReinhardTonemap(color: vec3<f32>) -> vec3<f32> {
  return color / (vec3<f32>(1.0) + color);
}

// ───── PUBLIC INTERFACE ─────

@vertex
fn vertexFullscreenTriangle(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var out: VertexOutput;
  let position = getFullscreenTrianglePosition(vertexIndex);
  out.position = vec4<f32>(position, 0.0, 1.0);
  out.uv = computeUvFromClipPosition(position);
  return out;
}

@fragment
fn fragmentExtract(in: VertexOutput) -> @location(0) vec4<f32> {
  let threshold = uniforms.thresholdIntensityExposure.x;
  let color = textureSample(sourceTexture, textureSampler, in.uv).rgb;
  let bright = extractBrightness(color, threshold);
  return vec4<f32>(bright, 1.0);
}

@fragment
fn fragmentBlur(in: VertexOutput) -> @location(0) vec4<f32> {
  const BLUR_WEIGHTS: array<f32, 9> = array<f32, 9>(0.0162, 0.0540, 0.1216, 0.1945, 0.2270, 0.1945, 0.1216, 0.0540, 0.0162);
  let texelSize = 1.0 / vec2<f32>(textureDimensions(sourceTexture));
  let step = uniforms.blurDirection.xy * texelSize;
  var sum = vec4<f32>(0.0);
  for (var k = 0; k < 9; k = k + 1) {
    let offset = f32(k) - 4.0;
    sum = sum + textureSample(sourceTexture, textureSampler, in.uv + step * offset) * BLUR_WEIGHTS[k];
  }
  return sum;
}

@fragment
fn fragmentComposite(in: VertexOutput) -> @location(0) vec4<f32> {
  let mainSample = textureSample(sourceTexture, textureSampler, in.uv);
  let bloom = textureSample(bloomTexture, textureSampler, in.uv).rgb;

  if (isBackgroundPixel(mainSample.a)) {
    return vec4<f32>(mainSample.rgb, 1.0);
  }

  let intensity = uniforms.thresholdIntensityExposure.y;
  let exposure = uniforms.thresholdIntensityExposure.z;
  let combined = combineBloom(mainSample.rgb, bloom, intensity, exposure);
  let tonemapped = applyReinhardTonemap(combined);
  return vec4<f32>(tonemapped, 1.0);
}
`;