import { getDensityFieldShaderChunk } from '../rendering/densityField.js';
import { getDripPhysicsShaderChunk } from './dripPhysics.js';
import { getDripPhaseSystemShaderChunk } from './dripPhaseSystem.js';

const UNIFORM_BUFFER_SIZE = 2 * 16;

// ───── WGSL SHADER ─────

export const SHADER_SOURCE = /* wgsl */ `
const GRAVITY: f32 = 1.8;
const GAMMA: f32 = 0.3;
const RESPAWN_OVERSHOOT: f32 = 1.5;

struct DripUniforms {
  hTNowDtFluidDensity: vec4<f32>,
  respawnYBaseRadiusDropCountAnchorRadius: vec4<f32>,
}

struct Drop {
  positionAndRadius: vec4<f32>,
  velocityAndSpeed: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: DripUniforms;
@group(0) @binding(1) var<storage, read> currentDrops: array<Drop>;
@group(0) @binding(2) var<storage, read> currentPairState: array<PairState>;
@group(0) @binding(3) var<storage, read_write> nextDrops: array<Drop>;
@group(0) @binding(4) var<storage, read_write> nextPairState: array<PairState>;

${getDensityFieldShaderChunk()}
${getDripPhysicsShaderChunk()}
${getDripPhaseSystemShaderChunk()}

@compute @workgroup_size(1)
fn computeMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let h = uniforms.hTNowDtFluidDensity.x;
  let tNow = uniforms.hTNowDtFluidDensity.y;
  let dt = uniforms.hTNowDtFluidDensity.z;
  let fluidDensity = uniforms.hTNowDtFluidDensity.w;
  let respawnY = uniforms.respawnYBaseRadiusDropCountAnchorRadius.x;
  let baseRadius = uniforms.respawnYBaseRadiusDropCountAnchorRadius.y;
  let dropCount = u32(uniforms.respawnYBaseRadiusDropCountAnchorRadius.z);
  let anchorBaseRadius = uniforms.respawnYBaseRadiusDropCountAnchorRadius.w;

  let i = globalId.x;
  if (i >= dropCount) {
    return;
  }

  let isDrip = (i % 2u) == 1u;
  if (!isDrip) {
    var next: Drop;
    next.positionAndRadius = vec4<f32>(currentDrops[i].positionAndRadius.xyz, anchorBaseRadius);
    next.velocityAndSpeed = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    nextDrops[i] = next;
    return;
  }

  let pairIndex = i / 2u;
  let anchorIndex = i - 1u;
  let pair = currentPairState[pairIndex];
  let anchorPosition = currentDrops[anchorIndex].positionAndRadius.xyz;

  // ───── DISPATCHER, WEIGHTS, BLENDING (the only phase-aware reads) ─────
  let weights = computePhaseWeights(pair, tNow, 1e-6);
  let gravity = blendPhaseValue(weights, getAttachedGravity(), getGrowingGravity(), getFallingGravity());
  let drag = blendPhaseValue(weights, getAttachedDrag(), getGrowingDrag(), getFallingDrag());

  // ───── FORCES (unconditional — no phase filtering, natural coalescing) ─────
  var force = vec3<f32>(0.0, 0.0, 0.0);
  let myPosition = currentDrops[i].positionAndRadius.xyz;
  let myMass = computeParticleMass(currentDrops[i].positionAndRadius.w, fluidDensity);
  for (var j = 0u; j < dropCount; j = j + 1u) {
    if (j == i) {
      continue;
    }
    let jPosition = currentDrops[j].positionAndRadius.xyz;
    let jMass = computeParticleMass(currentDrops[j].positionAndRadius.w, fluidDensity);
    force = force + computeCohesionForce(myPosition, jPosition, myMass, jMass, GAMMA, h);
  }

  let integrationMass = computeParticleMass(baseRadius, fluidDensity);
  let acceleration = force / integrationMass + vec3<f32>(0.0, -gravity, 0.0);

  let oldVelocity = currentDrops[i].velocityAndSpeed.xyz;
  let oldPosition = currentDrops[i].positionAndRadius.xyz;
  let newVelocity = oldVelocity * (1.0 - drag * dt) + acceleration * dt;
  let newPosition = oldPosition + newVelocity * dt;

  let separation = length(newPosition - anchorPosition);
  let nextPair = scheduleTick(pair, tNow, separation, newPosition.y, h, respawnY, pairIndex);

  var next: Drop;
  next.positionAndRadius = vec4<f32>(newPosition, baseRadius);
  next.velocityAndSpeed = vec4<f32>(newVelocity, length(newVelocity));

  if (getPhaseCode(nextPair) == PHASE_ATTACHED && getPhaseCode(pair) == PHASE_FALLING) {
    next.positionAndRadius = vec4<f32>(anchorPosition + vec3<f32>(0.0, RESPAWN_OVERSHOOT, 0.0), baseRadius);
    next.velocityAndSpeed = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }

  nextDrops[i] = next;
  nextPairState[pairIndex] = nextPair;
}
`;

// ───── PIPELINE SETUP ─────

export function makeDripComputePass(device) {
  const uniformBuffer = device.createBuffer({
    size: UNIFORM_BUFFER_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });

  const shaderModule = device.createShaderModule({ code: SHADER_SOURCE });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: shaderModule, entryPoint: 'computeMain' },
  });

  return { device, uniformBuffer, bindGroupLayout, pipeline };
}

export function writeDripPhysicsUniforms(computePass, view) {
  const data = new Float32Array(UNIFORM_BUFFER_SIZE / 4);
  data.set([view.h, view.tNow, view.dt, view.fluidDensity], 0);
  data.set([view.respawnY, view.baseRadius, view.dropCount, view.anchorBaseRadius], 4);
  computePass.device.queue.writeBuffer(computePass.uniformBuffer, 0, data);
}

export function makeDripComputeBindGroup(computePass, currentDropBuffer, currentPairStateBuffer, nextDropBuffer, nextPairStateBuffer) {
  return computePass.device.createBindGroup({
    layout: computePass.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: computePass.uniformBuffer } },
      { binding: 1, resource: { buffer: currentDropBuffer } },
      { binding: 2, resource: { buffer: currentPairStateBuffer } },
      { binding: 3, resource: { buffer: nextDropBuffer } },
      { binding: 4, resource: { buffer: nextPairStateBuffer } },
    ],
  });
}

export function runDripComputePass(computePass, commandEncoder, bindGroup, dropCount) {
  const pass = commandEncoder.beginComputePass();
  pass.setPipeline(computePass.pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(dropCount);
  pass.end();
}