import { registerBuffer, getBuffer } from '../core/resourceRegistry.js';
import { computeDensityKernel, computeParticleMass, computeRadiusFromMass } from '../rendering/densityField.js';

const PHASE_CODES = { ATTACHED: 0, GROWING: 1, FALLING: 2 };
const PHASE_NAMES = Object.fromEntries(Object.entries(PHASE_CODES).map(([name, code]) => [code, name]));

export function encodePhase(phase) {
  return PHASE_CODES[phase];
}

export function decodePhase(code) {
  return PHASE_NAMES[code];
}

export function makePairs(pairCount) {
  const pairs = [];
  for (let i = 0; i < pairCount; i += 1) {
    pairs.push({ anchorIndex: 2 * i, dripIndex: 2 * i + 1 });
  }
  return pairs;
}

export function computeInitialPhases(pairCount, activePairIndex) {
  return Array.from({ length: pairCount }, (_, i) => (i === activePairIndex ? 'GROWING' : 'ATTACHED'));
}

export function computePhaseTransition({ phase, separation, smoothingRadius }) {
  if (phase === 'GROWING' && separation > smoothingRadius) {
    return 'FALLING';
  }
  return phase;
}

export function computeRespawnTransition({ phase, dripY, respawnY }) {
  if (phase === 'FALLING' && dripY < respawnY) {
    return 'GROWING';
  }
  return phase;
}

export function computeCompensatedAnchorRadius({
  anchorBaseRadius,
  dripBaseRadius,
  separation,
  smoothingRadius,
  fluidDensity,
}) {
  const anchorBaseMass = computeParticleMass(anchorBaseRadius, fluidDensity);
  const dripBaseMass = computeParticleMass(dripBaseRadius, fluidDensity);
  const kernelRatio = computeDensityKernel(separation, smoothingRadius) / computeDensityKernel(0, smoothingRadius);
  const missingMass = dripBaseMass * (1 - kernelRatio);
  return computeRadiusFromMass(anchorBaseMass + missingMass, fluidDensity);
}

// ───── GPU BUFFER WRAPPER ─────

export function makePairState(device, registry, pairCount, initialPhases) {
  const size = pairCount * 4;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  registerBuffer(registry, 'pairStateA', { size, usage });
  registerBuffer(registry, 'pairStateB', { size, usage });
  const packed = new Float32Array(initialPhases.map(encodePhase));
  device.queue.writeBuffer(getBuffer(registry, 'pairStateA'), 0, packed);
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