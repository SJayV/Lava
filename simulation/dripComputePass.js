import { getDensityFieldShaderChunk } from '../rendering/densityField.js';
import { getDripPhysicsShaderChunk } from './dripPhysics.js';

const UNIFORM_BUFFER_SIZE = 3 * 16;

// ───── WGSL SHADER ─────

export const SHADER_SOURCE = /* wgsl */ `
const GRAVITY: f32 = 1.0;

struct DripUniforms {
  hGammaMu: vec4<f32>,
  dtFluidDensityRespawnY: vec4<f32>,
  dripRadiusDropCountPairCountAnchorRadius: vec4<f32>,
}

struct Drop {
  positionAndRadius: vec4<f32>,
  velocityAndSpeed: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: DripUniforms;
@group(0) @binding(1) var<storage, read> currentDrops: array<Drop>;
@group(0) @binding(2) var<storage, read> currentPairState: array<f32>;
@group(0) @binding(3) var<storage, read_write> nextDrops: array<Drop>;
@group(0) @binding(4) var<storage, read_write> nextPairState: array<f32>;

${getDensityFieldShaderChunk()}
${getDripPhysicsShaderChunk()}

@compute @workgroup_size(1)
fn computeMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let h = uniforms.hGammaMu.x;
  let gamma = uniforms.hGammaMu.y;
  let mu = uniforms.hGammaMu.z;
  let dt = uniforms.dtFluidDensityRespawnY.x;
  let fluidDensity = uniforms.dtFluidDensityRespawnY.y;
  let respawnY = uniforms.dtFluidDensityRespawnY.z;
  let baseRadius = uniforms.dripRadiusDropCountPairCountAnchorRadius.x;
  let dropCount = u32(uniforms.dripRadiusDropCountPairCountAnchorRadius.y);
  let anchorBaseRadius = uniforms.dripRadiusDropCountPairCountAnchorRadius.w;

  let i = globalId.x;
  if (i >= dropCount) {
    return;
  }

  let isDrip = (i % 2u) == 1u;
  if (!isDrip) {
    let dripIndex = i + 1u;
    let separation = length(currentDrops[dripIndex].positionAndRadius.xyz - currentDrops[i].positionAndRadius.xyz);
    let anchorBaseMass = computeParticleMass(anchorBaseRadius, fluidDensity);
    let dripBaseMass = computeParticleMass(baseRadius, fluidDensity);
    let kernelRatio = computeDensityKernel(separation, h) / computeDensityKernel(0.0, h);
    let missingMass = dripBaseMass * (1.0 - kernelRatio);
    let compensatedRadius = computeRadiusFromMass(anchorBaseMass + missingMass, fluidDensity);

    var next: Drop;
    next.positionAndRadius = vec4<f32>(currentDrops[i].positionAndRadius.xyz, compensatedRadius);
    next.velocityAndSpeed = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    nextDrops[i] = next;
    return;
  }

  let pairIndex = i / 2u;
  let anchorIndex = i - 1u;
  let phaseCode = currentPairState[pairIndex];
  let anchorPosition = currentDrops[anchorIndex].positionAndRadius.xyz;

  if (phaseCode == 0.0) {
    var next: Drop;
    next.positionAndRadius = vec4<f32>(anchorPosition, baseRadius);
    next.velocityAndSpeed = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    nextDrops[i] = next;
    nextPairState[pairIndex] = 0.0;
    return;
  }

  var force = vec3<f32>(0.0, 0.0, 0.0);
  if (phaseCode == 1.0) {
    let myPosition = currentDrops[i].positionAndRadius.xyz;
    let myMass = computeParticleMass(currentDrops[i].positionAndRadius.w, fluidDensity);
    for (var j = 0u; j < dropCount; j = j + 1u) {
      if (j == i) {
        continue;
      }
      let jIsDrip = (j % 2u) == 1u;
      if (jIsDrip) {
        let jPairIndex = j / 2u;
        if (currentPairState[jPairIndex] == 2.0) {
          continue;
        }
      }
      let jPosition = currentDrops[j].positionAndRadius.xyz;
      let jMass = computeParticleMass(currentDrops[j].positionAndRadius.w, fluidDensity);
      force = force + computeCohesionForce(myPosition, jPosition, myMass, jMass, gamma, h);
    }
  }

  let integrationMass = computeParticleMass(baseRadius, fluidDensity);
  let acceleration = force / integrationMass + vec3<f32>(0.0, -GRAVITY, 0.0);

  let oldVelocity = currentDrops[i].velocityAndSpeed.xyz;
  let oldPosition = currentDrops[i].positionAndRadius.xyz;
  let newVelocity = oldVelocity * (1.0 - mu * dt) + acceleration * dt;
  let newPosition = oldPosition + newVelocity * dt;

  if (phaseCode == 2.0 && newPosition.y < respawnY) {
    var resetDrop: Drop;
    resetDrop.positionAndRadius = vec4<f32>(anchorPosition, baseRadius);
    resetDrop.velocityAndSpeed = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    nextDrops[i] = resetDrop;
    nextPairState[pairIndex] = 1.0;
    return;
  }

  var newPhaseCode = phaseCode;
  if (phaseCode == 1.0) {
    let separation = length(newPosition - anchorPosition);
    if (separation > h) {
      newPhaseCode = 2.0;
    }
  }

  var next: Drop;
  next.positionAndRadius = vec4<f32>(newPosition, baseRadius);
  next.velocityAndSpeed = vec4<f32>(newVelocity, length(newVelocity));
  nextDrops[i] = next;
  nextPairState[pairIndex] = newPhaseCode;
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
  data.set([view.h, view.gamma, view.mu, 0], 0);
  data.set([view.dt, view.fluidDensity, view.respawnY, 0], 4);
  data.set([view.baseRadius, view.dropCount, view.pairCount, view.anchorBaseRadius], 8);
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