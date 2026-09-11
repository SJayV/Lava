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

// ───── BRIGHT-PASS EXTRACTION ─────

@fragment
fn fragmentExtract(in: VertexOutput) -> @location(0) vec4<f32> {
  let threshold = uniforms.thresholdIntensityExposure.x;
  let color = textureSample(sourceTexture, textureSampler, in.uv).rgb;
  let bright = max(color - vec3<f32>(threshold), vec3<f32>(0.0));
  return vec4<f32>(bright, 1.0);
}

// ───── SEPARABLE GAUSSIAN BLUR ─────

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

// ───── COMPOSITE ─────

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