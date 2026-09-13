import { getParticleMassChunk } from '../shaderChunks/shapeChunk.js';
import { getSimulationChunk } from '../shaderChunks/simulationChunk.js';
import { getPhaseChunk } from '../shaderChunks/phaseChunk.js';

export const SHADER_SOURCE = /* wgsl */ `
const GAMMA: f32 = 2.0;
const RESPAWN_OVERSHOOT: f32 = 1.5;
const GRAVITY: f32 = 5.0;
const BASE_DRAG: f32 = 9.0;
const FALLING_DRAG_FACTOR: f32 = 0.0005;

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

// ───── PHASE SCHEDULER: EXIT PREDICATES + ACTIVATION ─────

fn attachedShouldExit(pair: PairState, tNow: f32, pairIndex: u32) -> bool {
  let muAttached = getMus(pair).x;
  let gap = computeAttachedGrowingGap(pairIndex, muAttached);
  return tNow >= muAttached + gap;
}

fn growingShouldExit(separation: f32, h: f32) -> bool {
  return separation > h;
}

fn fallingShouldExit(dripY: f32, respawnY: f32) -> bool {
  return dripY < respawnY;
}

fn activateGrowing(pair: PairState, tNow: f32) -> PairState {
  var next = pair;
  next.phaseCodeAndMus.y = computeBumpDeactivationMu(tNow, SIGMA_ATTACHED);
  next.phaseCodeAndMus.z = computeBumpActivationMu(tNow, SIGMA_GROWING);
  next.phaseCodeAndMus.x = PHASE_GROWING;
  return next;
}

fn activateFalling(pair: PairState, tNow: f32) -> PairState {
  var next = pair;
  next.phaseCodeAndMus.z = computeBumpDeactivationMu(tNow, SIGMA_GROWING);
  next.phaseCodeAndMus.w = computeBumpActivationMu(tNow, SIGMA_FALLING);
  next.phaseCodeAndMus.x = PHASE_FALLING;
  return next;
}

fn activateAttached(pair: PairState, tNow: f32) -> PairState {
  var next = pair;
  next.phaseCodeAndMus.w = computeBumpDeactivationMu(tNow, SIGMA_FALLING);
  next.phaseCodeAndMus.y = computeBumpActivationMu(tNow, SIGMA_ATTACHED);
  next.phaseCodeAndMus.x = PHASE_ATTACHED;
  return next;
}

// ───── PHASE SCHEDULER: DISPATCHER + PER-PHASE HANDLERS ─────

fn scheduleAttached(pair: PairState, tNow: f32, pairIndex: u32) -> PairState {
  if (attachedShouldExit(pair, tNow, pairIndex)) {
    return activateGrowing(pair, tNow);
  }
  return pair;
}

fn scheduleGrowing(pair: PairState, tNow: f32, separation: f32, h: f32) -> PairState {
  var next = pair;
  next.phaseCodeAndMus.z = max(pair.phaseCodeAndMus.z, tNow);
  if (growingShouldExit(separation, h)) {
    return activateFalling(next, tNow);
  }
  return next;
}

fn scheduleFalling(pair: PairState, tNow: f32, dripY: f32, respawnY: f32) -> PairState {
  var next = pair;
  next.phaseCodeAndMus.w = max(pair.phaseCodeAndMus.w, tNow);
  if (fallingShouldExit(dripY, respawnY)) {
    return activateAttached(next, tNow);
  }
  return next;
}

fn scheduleTick(pair: PairState, tNow: f32, separation: f32, dripY: f32, h: f32, respawnY: f32, pairIndex: u32) -> PairState {
  let phaseCode = getPhaseCode(pair);
  if (phaseCode == PHASE_ATTACHED) {
    return scheduleAttached(pair, tNow, pairIndex);
  } else if (phaseCode == PHASE_GROWING) {
    return scheduleGrowing(pair, tNow, separation, h);
  } else {
    return scheduleFalling(pair, tNow, dripY, respawnY);
  }
}

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
  let gravity = blendPhaseValue(weights, 0.0, GRAVITY, GRAVITY);
  let drag = blendPhaseValue(weights, BASE_DRAG, BASE_DRAG, BASE_DRAG * FALLING_DRAG_FACTOR);

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