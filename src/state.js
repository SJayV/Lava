import { registerBuffer, getBuffer } from './gpuSetup.js';

// ───── DROP STATE ─────

export const LINE_Y = 1.2;

const FLOATS_PER_DROP = 8;
const BYTES_PER_DROP = FLOATS_PER_DROP * 4;

export function computeSmoothingRadiusFromLineSpan({ ballCount, lineSpanWidth, spacingToHRatio = 0.6 }) {
  const spacing = lineSpanWidth / (ballCount - 1);
  return spacing / spacingToHRatio;
}

export function computeLineSeedPositions({ ballCount, lineSpanWidth, lineY = 0 }) {
  const positions = [];
  for (let i = 0; i < ballCount; i += 1) {
    const x = -lineSpanWidth / 2 + (i / (ballCount - 1)) * lineSpanWidth;
    positions.push([x, lineY, 0]);
  }
  return positions;
}

export function makeDropRecord({ position, radius, velocity = [0, 0, 0] }) {
  return { position, radius, velocity };
}

export function packDropRecords(drops) {
  const packed = new Float32Array(drops.length * FLOATS_PER_DROP);
  drops.forEach((drop, index) => {
    const offset = index * FLOATS_PER_DROP;
    const [x, y, z] = drop.position;
    const [vx, vy, vz] = drop.velocity;
    const speed = Math.hypot(vx, vy, vz);
    packed.set([x, y, z, drop.radius, vx, vy, vz, speed], offset);
  });
  return packed;
}

export function makeDropState(registry, dropCount) {
  const size = dropCount * BYTES_PER_DROP;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  registerBuffer(registry, 'dropStateA', { size, usage });
  registerBuffer(registry, 'dropStateB', { size, usage });
  return { registry, dropCount, activeIndex: 0 };
}

export function getCurrentDropBuffer(dropState) {
  return getBuffer(dropState.registry, dropState.activeIndex === 0 ? 'dropStateA' : 'dropStateB');
}

export function swapDropState(dropState) {
  dropState.activeIndex = 1 - dropState.activeIndex;
}

// ───── PAIR STATE ─────

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

export function makePairState(device, registry, pairCount, initialPairStates) {
  const size = pairCount * FLOATS_PER_PAIR_STATE * 4;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  registerBuffer(registry, 'pairStateA', { size, usage });
  registerBuffer(registry, 'pairStateB', { size, usage });
  device.queue.writeBuffer(getBuffer(registry, 'pairStateA'), 0, packPairStateRecords(initialPairStates));
  return { registry, pairCount, activeIndex: 0 };
}

export function swapPairState(pairState) {
  pairState.activeIndex = 1 - pairState.activeIndex;
}