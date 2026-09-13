import { getParticleMassChunk } from '../shaderChunks/shapeChunk.js';
import { getSimulationChunk } from '../shaderChunks/simulationChunk.js';
import { getPhaseChunk } from '../shaderChunks/phaseChunk.js';

export const SHADER_SOURCE = /* wgsl */ `
const GAMMA: f32 = 2.0;
const RESPAWN_OVERSHOOT: f32 = 1.5;

struct DripUniforms {
  hTNowDtFluidDensity: vec4<f32>,
  respawnYBaseRadiusPairCount: vec4<f32>,
}

struct Drop {
  positionAndRadius: vec4<f32>,
  velocityAndSpeed: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: DripUniforms;
@group(0) @binding(1) var<storage, read> anchors: array<Drop>;
@group(0) @binding(2) var<storage, read> currentDrips: array<Drop>;
@group(0) @binding(3) var<storage, read> currentPairState: array<PairState>;
@group(0) @binding(4) var<storage, read_write> nextDrips: array<Drop>;
@group(0) @binding(5) var<storage, read_write> nextPairState: array<PairState>;

${getParticleMassChunk()}
${getSimulationChunk()}
${getPhaseChunk()}

@compute @workgroup_size(1)
fn computeMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let h = uniforms.hTNowDtFluidDensity.x;
  let tNow = uniforms.hTNowDtFluidDensity.y;
  let dt = uniforms.hTNowDtFluidDensity.z;
  let fluidDensity = uniforms.hTNowDtFluidDensity.w;
  let respawnY = uniforms.respawnYBaseRadiusPairCount.x;
  let baseRadius = uniforms.respawnYBaseRadiusPairCount.y;
  let pairCount = u32(uniforms.respawnYBaseRadiusPairCount.z);

  let i = globalId.x;
  if (i >= pairCount) {
    return;
  }

  let pair = currentPairState[i];
  let anchorPosition = anchors[i].positionAndRadius.xyz;

  // ───── DISPATCHER, WEIGHTS, BLENDING ─────
  let weights = computePhaseWeights(pair, tNow, 1e-6);
  let gravity = blendPhaseValue(weights, getAttachedGravity(), getGrowingGravity(), getFallingGravity());
  let drag = blendPhaseValue(weights, getAttachedDrag(), getGrowingDrag(), getFallingDrag());

  // ───── FORCES ─────
  let myPosition = currentDrips[i].positionAndRadius.xyz;
  let myMass = computeParticleMass(currentDrips[i].positionAndRadius.w, fluidDensity);
  let anchorMass = computeParticleMass(anchors[i].positionAndRadius.w, fluidDensity);
  let force = computeCohesionForce(myPosition, anchorPosition, myMass, anchorMass, GAMMA, h);

  let integrationMass = computeParticleMass(baseRadius, fluidDensity);
  let acceleration = force / integrationMass + vec3<f32>(0.0, -gravity, 0.0);

  let oldVelocity = currentDrips[i].velocityAndSpeed.xyz;
  let oldPosition = currentDrips[i].positionAndRadius.xyz;
  let newVelocity = oldVelocity * (1.0 - drag * dt) + acceleration * dt;
  let newPosition = oldPosition + newVelocity * dt;

  let separation = length(newPosition - anchorPosition);
  let nextPair = scheduleTick(pair, tNow, separation, newPosition.y, h, respawnY, i);

  var next: Drop;
  next.positionAndRadius = vec4<f32>(newPosition, baseRadius);
  next.velocityAndSpeed = vec4<f32>(newVelocity, length(newVelocity));

  if (getPhaseCode(nextPair) == PHASE_ATTACHED && getPhaseCode(pair) == PHASE_FALLING) {
    next.positionAndRadius = vec4<f32>(anchorPosition + vec3<f32>(0.0, RESPAWN_OVERSHOOT, 0.0), baseRadius);
    next.velocityAndSpeed = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }

  nextDrips[i] = next;
  nextPairState[i] = nextPair;
}
`;