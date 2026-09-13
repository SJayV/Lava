import { getHashChunk } from './noiseChunk.js';

export function getPhaseChunk() {
  return `
    ${getHashChunk()}

    // ───── CONSTANTS ─────

    const LEAD: f32 = 3.0;
    const SIGMA_ATTACHED: f32 = 1.5;
    const SIGMA_GROWING: f32 = 1.5;
    const SIGMA_FALLING: f32 = 0.92;

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

    // ───── PAIRSTATE ACCESS ─────

    fn getPhaseCode(pair: PairState) -> f32 {
      return pair.phaseCodeAndMus.x;
    }

    fn getMus(pair: PairState) -> vec3<f32> {
      return pair.phaseCodeAndMus.yzw;
    }

    // ───── GAUSSIAN WEIGHTING ─────

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
      const HOLD_MIN: f32 = 1.0;
      const HOLD_MAX: f32 = 14.5;
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
  `;
}