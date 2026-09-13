import { getParticleMassChunk } from '../shaderChunks/shapeChunk.js';
import { getSimulationChunk } from '../shaderChunks/simulationChunk.js';
import { getPhaseChunk } from '../shaderChunks/phaseChunk.js';

export const SHADER_SOURCE = `
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

// ───── DISPATCHER ─────

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

// ───── SCHEDULER ─────

fn scheduleAttached(pair: PairState, tNow: f32, pairIndex: u32) -> PairState {
  var next = pair;
  if (attachedShouldExit(pair, tNow, pairIndex)) {
    return activateGrowing(next, tNow);
  }
  return next;
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
  }
  if (phaseCode == PHASE_GROWING) {
    return scheduleGrowing(pair, tNow, separation, h);
  } else {
    return scheduleFalling(pair, tNow, dripY, respawnY);
  }
}

// ───── HELPER FUNCTIONS - PHASE BLENDING ─────

struct PhaseBlend {
  gravity: f32,
  drag: f32,
}

fn computePhaseBlend(pair: PairState, tNow: f32) -> PhaseBlend {
  const GRAVITY: f32 = 5.0;
  const BASE_DRAG: f32 = 9.0;
  const FALLING_DRAG_FACTOR: f32 = 0.0005;
  let weights = computePhaseWeights(pair, tNow, 1e-6);
  let gravity = blendPhaseValue(weights, 0.0, GRAVITY, GRAVITY);
  let drag = blendPhaseValue(weights, BASE_DRAG, BASE_DRAG, BASE_DRAG * FALLING_DRAG_FACTOR);
  return PhaseBlend(gravity, drag);
}

// ───── HELPER FUNCTIONS - FORCES ─────

fn computeNetAcceleration(myPosition: vec3<f32>, anchorPosition: vec3<f32>, myMass: f32, anchorMass: f32, integrationMass: f32, gravity: f32, h: f32) -> vec3<f32> {
  let force = computeCohesionForce(myPosition, anchorPosition, myMass, anchorMass, 2.0, h);
  return computeAcceleration(force, integrationMass, gravity);
}

// ───── HELPER FUNCTIONS - INTEGRATION ─────

fn computeAcceleration(force: vec3<f32>, integrationMass: f32, gravity: f32) -> vec3<f32> {
  return force / integrationMass + vec3<f32>(0.0, -gravity, 0.0);
}

fn integrateVelocity(oldVelocity: vec3<f32>, acceleration: vec3<f32>, drag: f32, dt: f32) -> vec3<f32> {
  return oldVelocity * (1.0 - drag * dt) + acceleration * dt;
}

fn integratePosition(oldPosition: vec3<f32>, velocity: vec3<f32>, dt: f32) -> vec3<f32> {
  return oldPosition + velocity * dt;
}

// ───── HELPER FUNCTIONS - RESPAWN ─────

fn justRespawned(nextPair: PairState, pair: PairState) -> bool {
  return getPhaseCode(nextPair) == PHASE_ATTACHED && getPhaseCode(pair) == PHASE_FALLING;
}

fn computeRespawnPosition(anchorPosition: vec3<f32>) -> vec3<f32> {
  const RESPAWN_OVERSHOOT: f32 = 1.5;
  return anchorPosition + vec3<f32>(0.0, RESPAWN_OVERSHOOT, 0.0);
}

fn buildNextDrop(newPosition: vec3<f32>, newVelocity: vec3<f32>, baseRadius: f32, nextPair: PairState, pair: PairState, anchorPosition: vec3<f32>) -> Drop {
  var next: Drop;
  next.positionAndRadius = vec4<f32>(newPosition, baseRadius);
  next.velocityAndSpeed = vec4<f32>(newVelocity, length(newVelocity));

  if (justRespawned(nextPair, pair)) {
    next.positionAndRadius = vec4<f32>(computeRespawnPosition(anchorPosition), baseRadius);
    next.velocityAndSpeed = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  }
  return next;
}

// ───── PUBLIC INTERFACE ─────

@compute @workgroup_size(1)
fn computeSimulationStep(@builtin(global_invocation_id) globalId: vec3<u32>) {
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

  let blend = computePhaseBlend(pair, tNow);

  let myPosition = currentDrips[i].positionAndRadius.xyz;
  let myMass = computeParticleMass(currentDrips[i].positionAndRadius.w, fluidDensity);
  let anchorMass = computeParticleMass(anchors[i].positionAndRadius.w, fluidDensity);
  let integrationMass = computeParticleMass(baseRadius, fluidDensity);
  let acceleration = computeNetAcceleration(myPosition, anchorPosition, myMass, anchorMass, integrationMass, blend.gravity, h);

  let oldVelocity = currentDrips[i].velocityAndSpeed.xyz;
  let oldPosition = currentDrips[i].positionAndRadius.xyz;
  let newVelocity = integrateVelocity(oldVelocity, acceleration, blend.drag, dt);
  let newPosition = integratePosition(oldPosition, newVelocity, dt);

  let separation = length(newPosition - anchorPosition);
  let nextPair = scheduleTick(pair, tNow, separation, newPosition.y, h, respawnY, i);

  nextDrips[i] = buildNextDrop(newPosition, newVelocity, baseRadius, nextPair, pair, anchorPosition);
  nextPairState[i] = nextPair;
}
`;