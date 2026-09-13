import { registerBuffer, getBuffer } from './gpuSetup.js';

// ───── CONSTANTS ─────

const FLOATS_PER_DROP = 8;
const BYTES_PER_DROP = FLOATS_PER_DROP * 4;
const FLOATS_PER_PAIR_STATE = 4;

// ───── HELPER FUNCTIONS - STORAGE BUFFERS ─────

function _initializeBuffer(registry, name, size) {
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  registerBuffer(registry, name, { size, usage });
}

function _writeBufferRecords(device, registry, name, records, packRecords) {
  device.queue.writeBuffer(getBuffer(registry, name), 0, packRecords(records));
}

function _initializePingPongBuffers(registry, baseName, size) {
  _initializeBuffer(registry, `${baseName}A`, size);
  _initializeBuffer(registry, `${baseName}B`, size);
}

function _initializePingPongState(device, registry, baseName, bufferSize, records, packRecords) {
  _initializePingPongBuffers(registry, baseName, bufferSize);
  _writeBufferRecords(device, registry, `${baseName}A`, records, packRecords);
  return { registry, activeIndex: 0 };
}

// ───── HELPER FUNCTIONS - DROP RECORDS ─────

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

function _buildLineDropRecords({ ballCount, lineSpanWidth, lineY, radius }) {
  const positions = computeLinePositions({ ballCount, lineSpanWidth, lineY });
  return positions.map((position) => initializeDropRecord({ position, radius }));
}

// ───── HELPER FUNCTIONS - DROP STATE BUFFERS ─────

function _initializeAnchorState(device, registry, { ballCount, lineSpanWidth, lineY, radius }) {
  const anchorRecords = _buildLineDropRecords({ ballCount, lineSpanWidth, lineY, radius });
  _initializeBuffer(registry, 'anchorState', ballCount * BYTES_PER_DROP);
  _writeBufferRecords(device, registry, 'anchorState', anchorRecords, packDropRecords);
  return { registry };
}

function _getAnchorBuffer(anchorState) {
  return getBuffer(anchorState.registry, 'anchorState');
}

function _initializeDropState(device, registry, { ballCount, lineSpanWidth, lineY, radius }) {
  const dropRecords = _buildLineDropRecords({ ballCount, lineSpanWidth, lineY, radius });
  const pingPongState = _initializePingPongState(device, registry, 'dropState', ballCount * BYTES_PER_DROP, dropRecords, packDropRecords);
  return { ...pingPongState, pairCount: ballCount };
}

export function getDropBufferPair(dropState) {
  return [getBuffer(dropState.registry, 'dropStateA'), getBuffer(dropState.registry, 'dropStateB')];
}

export function swapDropState(dropState) {
  dropState.activeIndex = 1 - dropState.activeIndex;
}

// ───── HELPER FUNCTIONS - PAIR STATE RECORDS ─────

export function initializePairState({ tNow = 0, startGrowing = false }) {
  const NEVER_TRIGGERED_MU = -1e9;
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

// ───── HELPER FUNCTIONS - PAIR STATE BUFFERS ─────

function _initializePairStateBuffers(device, registry, pairCount) {
  const initialPairStates = Array.from({ length: pairCount }, () => initializePairState({ tNow: Math.random() * 20 }));
  const pingPongState = _initializePingPongState(device, registry, 'pairState', pairCount * FLOATS_PER_PAIR_STATE * 4, initialPairStates, packPairStateRecords);
  return { ...pingPongState, pairCount };
}

export function getPairStateBufferPair(pairState) {
  return [getBuffer(pairState.registry, 'pairStateA'), getBuffer(pairState.registry, 'pairStateB')];
}

export function swapPairState(pairState) {
  pairState.activeIndex = 1 - pairState.activeIndex;
}

// ───── PUBLIC INTERFACE ─────

export function initializeSceneState(device, registry, { pairCount, lineSpanWidth, lineY, anchorRadius, dripRadius }) {
  const anchorState = _initializeAnchorState(device, registry, { ballCount: pairCount, lineSpanWidth, lineY, radius: anchorRadius });
  const dropState = _initializeDropState(device, registry, { ballCount: pairCount, lineSpanWidth, lineY, radius: dripRadius });
  const pairState = _initializePairStateBuffers(device, registry, pairCount);
  return { anchorBuffer: _getAnchorBuffer(anchorState), dropState, pairState };
}