import { getDensityKernelChunk, getParticleMassChunk } from '../shaderChunks/shapeChunk.js';
import { getNoiseChunk } from '../shaderChunks/noiseChunk.js';
import { getColorChunk } from '../shaderChunks/colorChunk.js';
import { FULLSCREEN_TRIANGLE_POSITION_CHUNK } from '../src/gpuSetup.js';

export const SHADER_SOURCE = /* wgsl */ `
struct RaymarchUniforms {
  cameraRight: vec4<f32>,
  cameraUp: vec4<f32>,
  cameraForward: vec4<f32>,
  cameraEye: vec4<f32>,
  resolutionAndCamera: vec4<f32>,
  traceBoundsAndMaxDist: vec4<f32>,
  fieldParams: vec4<f32>,
  stepParams: vec4<f32>,
  backgroundColor: vec4<f32>,
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

fn densityContribution(drop: Drop, position: vec3<f32>, h: f32, fluidDensity: f32) -> f32 {
  let distance = length(position - drop.positionAndRadius.xyz);
  let mass = computeParticleMass(drop.positionAndRadius.w, fluidDensity);
  return mass * computeDensityKernel(distance, h);
}

fn computeDensityField(position: vec3<f32>, relevantMask: u32) -> f32 {
  var total = 0.0;
  let pairCount = u32(uniforms.fieldParams.w);
  let h = uniforms.fieldParams.x;
  let fluidDensity = uniforms.fieldParams.z;
  for (var k = 0u; k < pairCount; k = k + 1u) {
    if ((relevantMask & (1u << k)) != 0u) {
      total = total + densityContribution(anchors[k], position, h, fluidDensity);
    }
    if ((relevantMask & (1u << (k + pairCount))) != 0u) {
      total = total + densityContribution(drips[k], position, h, fluidDensity);
    }
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

fn computeFieldGradientNormal(position: vec3<f32>, relevantMask: u32) -> vec3<f32> {
  let epsilon = 0.002;
  let dx = vec3<f32>(epsilon, 0.0, 0.0);
  let dy = vec3<f32>(0.0, epsilon, 0.0);
  let dz = vec3<f32>(0.0, 0.0, epsilon);
  let gradient = vec3<f32>(
    computeDensityField(position + dx, relevantMask) - computeDensityField(position - dx, relevantMask),
    computeDensityField(position + dy, relevantMask) - computeDensityField(position - dy, relevantMask),
    computeDensityField(position + dz, relevantMask) - computeDensityField(position - dz, relevantMask),
  );
  return -normalize(gradient);
}

fn computeTraceBoundsIntersection(rayOrigin: vec3<f32>, rayDirection: vec3<f32>) -> vec2<f32> {
  let halfExtents = uniforms.traceBoundsAndMaxDist.xyz;
  let inv = 1.0 / rayDirection;
  let lower = (-halfExtents - rayOrigin) * inv;
  let upper = (halfExtents - rayOrigin) * inv;
  let tNear = max(max(min(lower.x, upper.x), min(lower.y, upper.y)), min(lower.z, upper.z));
  let tFar = min(min(max(lower.x, upper.x), max(lower.y, upper.y)), max(lower.z, upper.z));
  return vec2<f32>(max(tNear, 0.0), min(tFar, uniforms.traceBoundsAndMaxDist.w));
}

fn computeRaySphereEntryExit(rayOrigin: vec3<f32>, rayDirection: vec3<f32>, sphereCenter: vec3<f32>, sphereRadius: f32) -> vec2<f32> {
  let offset = rayOrigin - sphereCenter;
  let b = dot(offset, rayDirection);
  let c = dot(offset, offset) - sphereRadius * sphereRadius;
  let discriminant = b * b - c;
  if (discriminant < 0.0) {
    return vec2<f32>(1.0, -1.0);
  }
  let sqrtDiscriminant = sqrt(discriminant);
  return vec2<f32>(-b - sqrtDiscriminant, -b + sqrtDiscriminant);
}

struct TraceResult {
  hit: bool,
  position: vec3<f32>,
  relevantMask: u32,
}

fn traceDensityIsosurface(rayOrigin: vec3<f32>, rayDirection: vec3<f32>) -> TraceResult {
  let bounds = computeTraceBoundsIntersection(rayOrigin, rayDirection);
  var result: TraceResult;
  result.hit = false;
  if (bounds.y <= bounds.x) {
    return result;
  }

  let h = uniforms.fieldParams.x;
  let pairCount = u32(uniforms.fieldParams.w);

  var clusterNear = bounds.y;
  var clusterFar = bounds.x;
  var anyBodyHit = false;
  var relevantMask = 0u;
  for (var k = 0u; k < pairCount; k = k + 1u) {
    let anchorInterval = computeRaySphereEntryExit(rayOrigin, rayDirection, anchors[k].positionAndRadius.xyz, h);
    if (anchorInterval.x <= anchorInterval.y) {
      anyBodyHit = true;
      relevantMask = relevantMask | (1u << k);
      clusterNear = min(clusterNear, max(anchorInterval.x, bounds.x));
      clusterFar = max(clusterFar, min(anchorInterval.y, bounds.y));
    }

    let dripInterval = computeRaySphereEntryExit(rayOrigin, rayDirection, drips[k].positionAndRadius.xyz, h);
    if (dripInterval.x <= dripInterval.y) {
      anyBodyHit = true;
      relevantMask = relevantMask | (1u << (k + pairCount));
      clusterNear = min(clusterNear, max(dripInterval.x, bounds.x));
      clusterFar = max(clusterFar, min(dripInterval.y, bounds.y));
    }
  }
  if (!anyBodyHit) {
    return result;
  }
  result.relevantMask = relevantMask;

  let isoLevel = uniforms.fieldParams.y;
  let minStep = uniforms.stepParams.x;
  let maxStep = uniforms.stepParams.y;
  let maxTraceSteps = u32(uniforms.stepParams.z);

  var t = clusterNear;
  var previousT = t;
  var previousSign = sign(computeDensityField(rayOrigin + rayDirection * t, relevantMask) - isoLevel);

  for (var i = 0u; i < maxTraceSteps; i = i + 1u) {
    let position = rayOrigin + rayDirection * t;
    let density = computeDensityField(position, relevantMask);
    let currentSign = sign(density - isoLevel);

    if (currentSign != previousSign) {
      var lo = previousT;
      var hi = t;
      for (var b = 0u; b < 6u; b = b + 1u) {
        let mid = 0.5 * (lo + hi);
        let midDensity = computeDensityField(rayOrigin + rayDirection * mid, relevantMask);
        if (sign(midDensity - isoLevel) == previousSign) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      result.hit = true;
      result.position = rayOrigin + rayDirection * (0.5 * (lo + hi));
      return result;
    }

    previousT = t;
    previousSign = currentSign;

    let step = clamp(abs(density - isoLevel) / uniforms.backgroundColor.w, minStep, maxStep);
    t = t + step;

    if (t >= clusterFar) {
      return result;
    }
  }

  return result;
}

${FULLSCREEN_TRIANGLE_POSITION_CHUNK}

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4<f32> {
  return vec4<f32>(getFullscreenTrianglePosition(vertexIndex), 0.0, 1.0);
}

@fragment
fn fragmentMain(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
  let width = uniforms.resolutionAndCamera.x;
  let height = uniforms.resolutionAndCamera.y;
  let aspectRatio = uniforms.resolutionAndCamera.z;
  let focalLength = uniforms.resolutionAndCamera.w;

  let ndcX = 2.0 * (fragCoord.x / width) - 1.0;
  let ndcY = 1.0 - 2.0 * (fragCoord.y / height);
  let viewSpaceDirection = vec3<f32>(ndcX * aspectRatio / focalLength, ndcY / focalLength, -1.0);
  let rayDirection = normalize(
    uniforms.cameraRight.xyz * viewSpaceDirection.x +
    uniforms.cameraUp.xyz * viewSpaceDirection.y +
    uniforms.cameraForward.xyz * viewSpaceDirection.z
  );

  let result = traceDensityIsosurface(uniforms.cameraEye.xyz, rayDirection);
  if (!result.hit) {
    return vec4<f32>(uniforms.backgroundColor.xyz, 0.0);
  }

  let noiseScale = uniforms.noiseParams.x;
  let noiseSpeed = uniforms.noiseParams.y;
  let noiseOctaves = u32(uniforms.noiseParams.z);
  let animationTime = uniforms.noiseParams.w;
  let heatValue = computeTurbulence(result.position, animationTime, noiseOctaves, noiseScale, noiseSpeed);

  let lavaColor = computeTemperatureColor(heatValue);

  let normal = computeFieldGradientNormal(result.position, result.relevantMask);
  let viewDirection = normalize(uniforms.cameraEye.xyz - result.position);
  let shadedColor = computeShadedColor(lavaColor, normal, viewDirection);

  return vec4<f32>(shadedColor, 1.0);
}
`;
