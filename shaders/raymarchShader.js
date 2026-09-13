import { getDensityKernelChunk, getParticleMassChunk } from '../shaderChunks/shapeChunk.js';
import { getNoiseChunk } from '../shaderChunks/noiseChunk.js';
import { getColorChunk } from '../shaderChunks/colorChunk.js';
import { FULLSCREEN_TRIANGLE_POSITION_CHUNK } from '../src/gpuSetup.js';

export const SHADER_SOURCE = `
struct RaymarchUniforms {
  cameraRight: vec4<f32>,
  cameraUp: vec4<f32>,
  cameraForward: vec4<f32>,
  cameraEye: vec4<f32>,
  resolutionAndCamera: vec4<f32>,
  traceBoundsAndMaxDist: vec4<f32>,
  fieldParams: vec4<f32>,
  stepParams: vec4<f32>,
  backgroundColorAndGradientBound: vec4<f32>,
  noiseParams: vec4<f32>,
}

struct Drop {
  positionAndRadius: vec4<f32>,
  velocityAndSpeed: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: RaymarchUniforms;
@group(0) @binding(1) var<storage, read> anchors: array<Drop>;
@group(0) @binding(2) var<storage, read> drips: array<Drop>;

${getDensityKernelChunk()}
${getParticleMassChunk()}
${getNoiseChunk()}
${getColorChunk()}

// ───── HELPER FUNCTIONS - DENSITY FIELD ─────

fn densityContribution(drop: Drop, position: vec3<f32>, h: f32, fluidDensity: f32) -> f32 {
  let distance = length(position - drop.positionAndRadius.xyz);
  let mass = computeParticleMass(drop.positionAndRadius.w, fluidDensity);
  return mass * computeDensityKernel(distance, h);
}

fn computeDensityField(position: vec3<f32>) -> f32 {
  var total = 0.0;
  let pairCount = u32(uniforms.fieldParams.w);
  let h = uniforms.fieldParams.x;
  let fluidDensity = uniforms.fieldParams.z;
  for (var k = 0u; k < pairCount; k = k + 1u) {
    total = total + densityContribution(anchors[k], position, h, fluidDensity);
    total = total + densityContribution(drips[k], position, h, fluidDensity);
  }
  return applySurfacePerturbation(total, position);
}

fn applySurfacePerturbation(density: f32, position: vec3<f32>) -> f32 {
  const BETA: f32 = 5.2;
  const PERTURBATION_FREQUENCY: f32 = 0.8;
  const PERTURBATION_SPEED: f32 = 0.4;
  let time = uniforms.noiseParams.w;
  let scaled = position * PERTURBATION_FREQUENCY + vec3<f32>(0.0, 0.0, time * PERTURBATION_SPEED);
  return density + BETA * computeGradientNoise3D(scaled);
}

fn computeFieldGradientNormal(position: vec3<f32>) -> vec3<f32> {
  const EPSILON: f32 = 0.002;
  let dx = vec3<f32>(EPSILON, 0.0, 0.0);
  let dy = vec3<f32>(0.0, EPSILON, 0.0);
  let dz = vec3<f32>(0.0, 0.0, EPSILON);
  let gradient = vec3<f32>(
    computeDensityField(position + dx) - computeDensityField(position - dx),
    computeDensityField(position + dy) - computeDensityField(position - dy),
    computeDensityField(position + dz) - computeDensityField(position - dz),
  );
  return -normalize(gradient);
}

// ───── HELPER FUNCTIONS - RAYMARCHING ─────

fn computeTraceBoundsIntersection(rayOrigin: vec3<f32>, rayDirection: vec3<f32>) -> vec2<f32> {
  let halfExtents = uniforms.traceBoundsAndMaxDist.xyz;
  let inv = 1.0 / rayDirection;
  let lower = (-halfExtents - rayOrigin) * inv;
  let upper = (halfExtents - rayOrigin) * inv;
  let tNear = max(max(min(lower.x, upper.x), min(lower.y, upper.y)), min(lower.z, upper.z));
  let tFar = min(min(max(lower.x, upper.x), max(lower.y, upper.y)), max(lower.z, upper.z));
  return vec2<f32>(max(tNear, 0.0), min(tFar, uniforms.traceBoundsAndMaxDist.w));
}

struct TraceResult {
  hit: bool,
  position: vec3<f32>,
}

fn missesTraceBounds(bounds: vec2<f32>) -> bool {
  return bounds.y <= bounds.x;
}

fn computeAdaptiveStep(density: f32, isoLevel: f32, minStep: f32, maxStep: f32) -> f32 {
  return clamp(abs(density - isoLevel) / uniforms.backgroundColorAndGradientBound.w, minStep, maxStep);
}

fn refineHitPosition(rayOrigin: vec3<f32>, rayDirection: vec3<f32>, previousT: f32, t: f32, previousSign: f32, isoLevel: f32) -> vec3<f32> {
  const BISECTION_STEPS: u32 = 6u;
  var low = previousT;
  var high = t;
  for (var b = 0u; b < BISECTION_STEPS; b = b + 1u) {
    let mid = 0.5 * (low + high);
    let midDensity = computeDensityField(rayOrigin + rayDirection * mid);
    if (sign(midDensity - isoLevel) == previousSign) {
      low = mid;
    } else {
      high = mid;
    }
  }
  return rayOrigin + rayDirection * (0.5 * (low + high));
}

fn traceDensityIsosurface(rayOrigin: vec3<f32>, rayDirection: vec3<f32>) -> TraceResult {
  let bounds = computeTraceBoundsIntersection(rayOrigin, rayDirection);
  var result: TraceResult;
  result.hit = false;
  if (missesTraceBounds(bounds)) {
    return result;
  }

  let isoLevel = uniforms.fieldParams.y;
  let minStep = uniforms.stepParams.x;
  let maxStep = uniforms.stepParams.y;
  let maxTraceSteps = u32(uniforms.stepParams.z);

  var t = bounds.x;
  var previousT = t;
  var previousSign = sign(computeDensityField(rayOrigin + rayDirection * t) - isoLevel);

  for (var i = 0u; i < maxTraceSteps; i = i + 1u) {
    let position = rayOrigin + rayDirection * t;
    let density = computeDensityField(position);
    let currentSign = sign(density - isoLevel);

    if (currentSign != previousSign) {
      result.hit = true;
      result.position = refineHitPosition(rayOrigin, rayDirection, previousT, t, previousSign, isoLevel);
      return result;
    }

    previousT = t;
    previousSign = currentSign;

    t = t + computeAdaptiveStep(density, isoLevel, minStep, maxStep);

    if (t >= bounds.y) {
      return result;
    }
  }

  return result;
}

${FULLSCREEN_TRIANGLE_POSITION_CHUNK}

// ───── HELPER FUNCTIONS - CAMERA RAY GENERATION ─────

fn computeScreenNdc(fragCoord: vec4<f32>, width: f32, height: f32) -> vec2<f32> {
  let ndcX = 2.0 * (fragCoord.x / width) - 1.0;
  let ndcY = 1.0 - 2.0 * (fragCoord.y / height);
  return vec2<f32>(ndcX, ndcY);
}

fn computeViewSpaceRayDirection(ndc: vec2<f32>, aspectRatio: f32, focalLength: f32) -> vec3<f32> {
  return vec3<f32>(ndc.x * aspectRatio / focalLength, ndc.y / focalLength, -1.0);
}

fn computeWorldRayDirection(cameraRight: vec3<f32>, cameraUp: vec3<f32>, cameraForward: vec3<f32>, viewSpaceDirection: vec3<f32>) -> vec3<f32> {
  return normalize(cameraRight * viewSpaceDirection.x + cameraUp * viewSpaceDirection.y + cameraForward * viewSpaceDirection.z);
}

// ───── HELPER FUNCTIONS - LAVA SHADING ─────

fn computeLavaColor(position: vec3<f32>, noiseParams: vec4<f32>) -> vec3<f32> {
  let heatValue = computeTurbulence(position, noiseParams.w, u32(noiseParams.z), noiseParams.x, noiseParams.y);
  return computeTemperatureColor(heatValue);
}

// ───── PUBLIC INTERFACE ─────

@vertex
fn vertexRenderScene(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
  return vec4<f32>(getFullscreenTrianglePosition(vertexIndex), 0.0, 1.0);
}

@fragment
fn fragmentRenderScene(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
  let ndc = computeScreenNdc(fragCoord, uniforms.resolutionAndCamera.x, uniforms.resolutionAndCamera.y);
  let viewSpaceDirection = computeViewSpaceRayDirection(ndc, uniforms.resolutionAndCamera.z, uniforms.resolutionAndCamera.w);
  let rayDirection = computeWorldRayDirection(uniforms.cameraRight.xyz, uniforms.cameraUp.xyz, uniforms.cameraForward.xyz, viewSpaceDirection);

  let result = traceDensityIsosurface(uniforms.cameraEye.xyz, rayDirection);
  if (!result.hit) {
    return vec4<f32>(uniforms.backgroundColorAndGradientBound.xyz, 0.0);
  }

  let lavaColor = computeLavaColor(result.position, uniforms.noiseParams);

  let normal = computeFieldGradientNormal(result.position);
  let viewDirection = normalize(uniforms.cameraEye.xyz - result.position);
  let shadedColor = computeShadedColor(lavaColor, normal, viewDirection);

  return vec4<f32>(shadedColor, 1.0);
}
`;