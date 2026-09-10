import { registerBuffer, getBuffer } from '../core/resourceRegistry.js';
import { getTurbulenceNoiseShaderChunk } from '../rendering/noise.js';

const FLOATS_PER_PAIR_STATE = 4;
const NEVER_TRIGGERED_MU = -1e9;

export function initializePairState({ tNow = 0, startGrowing = false }) {
  if (startGrowing) {
    return { phaseCode: 1, muAttached: NEVER_TRIGGERED_MU, muGrowing: tNow, muFalling: NEVER_TRIGGERED_MU };
  }
  return { phaseCode: 0, muAttached: tNow, muGrowing: NEVER_TRIGGERED_MU, muFalling: NEVER_TRIGGERED_MU };
}

export function makePairStateRecord({ phaseCode, muAttached, muGrowing, muFalling }) {
  return [phaseCode, muAttached, muGrowing, muFalling];
}

export function packPairStateRecords(records) {
  const packed = new Float32Array(records.length * FLOATS_PER_PAIR_STATE);
  records.forEach((record, index) => {
    packed.set(makePairStateRecord(record), index * FLOATS_PER_PAIR_STATE);
  });
  return packed;
}

// ───── GPU BUFFER WRAPPER ─────

export function makePairState(device, registry, pairCount, initialPairStates) {
  const size = pairCount * FLOATS_PER_PAIR_STATE * 4;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  registerBuffer(registry, 'pairStateA', { size, usage });
  registerBuffer(registry, 'pairStateB', { size, usage });
  device.queue.writeBuffer(getBuffer(registry, 'pairStateA'), 0, packPairStateRecords(initialPairStates));
  return { registry, pairCount, activeIndex: 0 };
}

export function getCurrentPairStateBuffer(pairState) {
  return getBuffer(pairState.registry, pairState.activeIndex === 0 ? 'pairStateA' : 'pairStateB');
}

export function getNextPairStateBuffer(pairState) {
  return getBuffer(pairState.registry, pairState.activeIndex === 0 ? 'pairStateB' : 'pairStateA');
}

export function swapPairState(pairState) {
  pairState.activeIndex = 1 - pairState.activeIndex;
}

// ───── WGSL CHUNK ─────

export function getDripPhaseSystemShaderChunk() {
  return /* wgsl */ `
    ${getTurbulenceNoiseShaderChunk()}

    // ───── CONSTANTS ─────

    const LEAD: f32 = 3.0;
    const SIGMA_ATTACHED: f32 = 1.5;
    const SIGMA_GROWING: f32 = 0.12;
    const SIGMA_FALLING: f32 = 0.12;
    const HOLD_MIN: f32 = 4.0;
    const HOLD_MAX: f32 = 24.5;
    const BASE_DRAG: f32 = 9.0;
    const FALLING_DRAG_FACTOR: f32 = 0.0005;

    const PHASE_ATTACHED: f32 = 0.0;
    const PHASE_GROWING: f32 = 1.0;
    const PHASE_FALLING: f32 = 2.0;

    struct PairState {
      phaseCodeAndMus: vec4<f32>,
    }

    struct PhaseWeights {
      wAttached: f32,
      wGrowing: f32,
      wFalling: f32,
    }

    // ───── PAIRSTATE ACCESS (unpack once, not per call site) ─────

    fn getPhaseCode(pair: PairState) -> f32 {
      return pair.phaseCodeAndMus.x;
    }

    fn getMus(pair: PairState) -> vec3<f32> {
      return pair.phaseCodeAndMus.yzw;
    }

    // ───── GAUSSIAN BUMP, WEIGHTS, BLENDING ─────

    fn computeGaussianBump(mu: f32, tNow: f32, sigma: f32) -> f32 {
      let d = tNow - mu;
      return exp(-(d * d) / (2.0 * sigma * sigma));
    }

    fn computeBumpActivationMu(triggerTime: f32, sigma: f32) -> f32 {
      return triggerTime + LEAD * sigma;
    }

    fn computeBumpDeactivationMu(triggerTime: f32, sigma: f32) -> f32 {
      return triggerTime - LEAD * sigma;
    }

    fn computeAttachedGrowingGap(pairIndex: u32, entryTime: f32) -> f32 {
      let t = hashLattice3D(i32(pairIndex), i32(entryTime * 1000.0), 0);
      return HOLD_MIN + t * (HOLD_MAX - HOLD_MIN);
    }

    fn computePhaseWeights(pair: PairState, tNow: f32, epsilon: f32) -> PhaseWeights {
      let mus = getMus(pair);
      let rawAttached = computeGaussianBump(mus.x, tNow, SIGMA_ATTACHED);
      let rawGrowing = computeGaussianBump(mus.y, tNow, SIGMA_GROWING);
      let rawFalling = computeGaussianBump(mus.z, tNow, SIGMA_FALLING);
      let sum = rawAttached + rawGrowing + rawFalling + epsilon;
      return PhaseWeights(rawAttached / sum, rawGrowing / sum, rawFalling / sum);
    }

    fn blendPhaseValue(weights: PhaseWeights, valueAttached: f32, valueGrowing: f32, valueFalling: f32) -> f32 {
      return valueAttached * weights.wAttached + valueGrowing * weights.wGrowing + valueFalling * weights.wFalling;
    }

    // ───── PER-PHASE PARAMETER VALUES (one function per phase — mirrors _clusterVelocity/_metaballVelocity/_burstVelocity; swap any one body for a derived formula later without touching call sites) ─────

    fn getAttachedGravity() -> f32 {
      return 0.0;
    }

    fn getGrowingGravity() -> f32 {
      return GRAVITY;
    }

    fn getFallingGravity() -> f32 {
      return GRAVITY;
    }

    fn getAttachedDrag() -> f32 {
      return BASE_DRAG;
    }

    fn getGrowingDrag() -> f32 {
      return BASE_DRAG;
    }

    fn getFallingDrag() -> f32 {
      return BASE_DRAG * FALLING_DRAG_FACTOR;
    }

    // ───── SCHEDULER: EXIT PREDICATES + ACTIVATION (cause before effect — a phase's mu is only ever written from inside its own activate<Phase> call, itself only ever reached through the matching <phase>ShouldExit check) ─────

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

    // ───── SCHEDULER: DISPATCHER + PER-PHASE HANDLERS ─────

    fn scheduleAttached(pair: PairState, tNow: f32, pairIndex: u32) -> PairState {
      if (attachedShouldExit(pair, tNow, pairIndex)) {
        return activateGrowing(pair, tNow);
      }
      return pair;
    }

    fn scheduleGrowing(pair: PairState, tNow: f32, separation: f32, h: f32) -> PairState {
      var next = pair;
      next.phaseCodeAndMus.z = tNow;
      if (growingShouldExit(separation, h)) {
        return activateFalling(next, tNow);
      }
      return next;
    }

    fn scheduleFalling(pair: PairState, tNow: f32, dripY: f32, respawnY: f32) -> PairState {
      var next = pair;
      next.phaseCodeAndMus.w = tNow;
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
  `;
}