import { registerBuffer, getBuffer } from './gpuSetup.js';

// ───── HELPER FUNCTIONS - PING-PONG BUFFERS ─────

function initializePingPongBuffers(registry, baseName, size) {
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  registerBuffer(registry, `${baseName}A`, { size, usage });
  registerBuffer(registry, `${baseName}B`, { size, usage });
}

// ───── DROP STATE ─────

const FLOATS_PER_DROP = 8;
const BYTES_PER_DROP = FLOATS_PER_DROP * 4;

export function computeSmoothingRadiusFromLineSpan({ ballCount, lineSpanWidth, spacingToHRatio = 0.6 }) {
  const spacing = lineSpanWidth / (ballCount - 1);
  return spacing / spacingToHRatio;
}

export function computeLinePositions({ ballCount, lineSpanWidth, lineY = 0 }) {
  const positions = [];
  for (let i = 0; i < ballCount; i += 1) {
    const x = -lineSpanWidth / 2 + (i / (ballCount - 1)) * lineSpanWidth;
    positions.push([x, lineY, 0]);
  }
  return positions;
}

export function initializeDropRecord({ position, radius, velocity = [0, 0, 0] }) {
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

export function initializeAnchorState(device, registry, { ballCount, lineSpanWidth, lineY, radius }) {
  const positions = computeLinePositions({ ballCount, lineSpanWidth, lineY });
  const anchorRecords = positions.map((position) => initializeDropRecord({ position, radius }));

  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  registerBuffer(registry, 'anchorState', { size: ballCount * BYTES_PER_DROP, usage });
  device.queue.writeBuffer(getBuffer(registry, 'anchorState'), 0, packDropRecords(anchorRecords));
  return { registry };
}

export function getAnchorBuffer(anchorState) {
  return getBuffer(anchorState.registry, 'anchorState');
}

export function initializeDripState(device, registry, { ballCount, lineSpanWidth, lineY, radius }) {
  const positions = computeLinePositions({ ballCount, lineSpanWidth, lineY });
  const dripRecords = positions.map((position) => initializeDropRecord({ position, radius }));

  initializePingPongBuffers(registry, 'dripState', ballCount * BYTES_PER_DROP);
  const dripState = { registry, pairCount: ballCount, activeIndex: 0 };
  device.queue.writeBuffer(getCurrentDripBuffer(dripState), 0, packDropRecords(dripRecords));
  return dripState;
}

export function getCurrentDripBuffer(dripState) {
  return getBuffer(dripState.registry, dripState.activeIndex === 0 ? 'dripStateA' : 'dripStateB');
}

export function getDripBufferPair(dripState) {
  return [getBuffer(dripState.registry, 'dripStateA'), getBuffer(dripState.registry, 'dripStateB')];
}

export function swapDripState(dripState) {
  dripState.activeIndex = 1 - dripState.activeIndex;
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

export function initializePairStateRecord({ phaseCode, muAttached, muGrowing, muFalling }) {
  return [phaseCode, muAttached, muGrowing, muFalling];
}

export function packPairStateRecords(records) {
  const packed = new Float32Array(records.length * FLOATS_PER_PAIR_STATE);
  records.forEach((record, index) => {
    packed.set(initializePairStateRecord(record), index * FLOATS_PER_PAIR_STATE);
  });
  return packed;
}

export function initializePairStateBuffers(device, registry, pairCount) {
  const initialPairStates = Array.from({ length: pairCount }, () => initializePairState({ tNow: Math.random() * 20 }));

  initializePingPongBuffers(registry, 'pairState', pairCount * FLOATS_PER_PAIR_STATE * 4);
  device.queue.writeBuffer(getBuffer(registry, 'pairStateA'), 0, packPairStateRecords(initialPairStates));
  return { registry, pairCount, activeIndex: 0 };
}

export function getPairStateBufferPair(pairState) {
  return [getBuffer(pairState.registry, 'pairStateA'), getBuffer(pairState.registry, 'pairStateB')];
}

export function swapPairState(pairState) {
  pairState.activeIndex = 1 - pairState.activeIndex;
}