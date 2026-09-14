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
  return { registry, baseName, activeIndex: 0 };
}

function _getPingPongBufferPair(pingPongState) {
  return [getBuffer(pingPongState.registry, `${pingPongState.baseName}A`), getBuffer(pingPongState.registry, `${pingPongState.baseName}B`)];
}

function _swapPingPongState(pingPongState) {
  pingPongState.activeIndex = 1 - pingPongState.activeIndex;
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

export function initializeDropRecordArray({ position, radius, velocity }) {
  const [x, y, z] = position;
  const [vx, vy, vz] = velocity;
  const speed = Math.hypot(vx, vy, vz);
  return [x, y, z, radius, vx, vy, vz, speed];
}

export function packDropRecords(drops) {
  const packed = new Float32Array(drops.length * FLOATS_PER_DROP);
  drops.forEach((drop, index) => {
    packed.set(initializeDropRecordArray(drop), index * FLOATS_PER_DROP);
  });
  return packed;
}

function _buildLineDropRecords({ ballCount, lineSpanWidth, lineY, radius }) {
  const positions = computeLinePositions({ ballCount, lineSpanWidth, lineY });
  return positions.map((position) => initializeDropRecord({ position, radius }));
}

// ───── HELPER FUNCTIONS - DROP STATE BUFFERS ─────

function _initializeAnchorBuffer(device, registry, { ballCount, lineSpanWidth, lineY, radius }) {
  const ANCHOR_BUFFER_NAME = 'anchorState';
  const anchorRecords = _buildLineDropRecords({ ballCount, lineSpanWidth, lineY, radius });
  _initializeBuffer(registry, ANCHOR_BUFFER_NAME, ballCount * BYTES_PER_DROP);
  _writeBufferRecords(device, registry, ANCHOR_BUFFER_NAME, anchorRecords, packDropRecords);
  return getBuffer(registry, ANCHOR_BUFFER_NAME);
}

function _initializeDropState(device, registry, { ballCount, lineSpanWidth, lineY, radius }) {
  const dropRecords = _buildLineDropRecords({ ballCount, lineSpanWidth, lineY, radius });
  const pingPongState = _initializePingPongState(device, registry, 'dropState', ballCount * BYTES_PER_DROP, dropRecords, packDropRecords);
  return { ...pingPongState, pairCount: ballCount };
}

export function getDropBufferPair(dropState) {
  return _getPingPongBufferPair(dropState);
}

export function swapDropState(dropState) {
  _swapPingPongState(dropState);
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
  return _getPingPongBufferPair(pairState);
}

export function swapPairState(pairState) {
  _swapPingPongState(pairState);
}

// ───── PUBLIC INTERFACE ─────

export function initializeSceneState(device, registry, { pairCount, lineSpanWidth, lineY, radius }) {
  const anchorBuffer = _initializeAnchorBuffer(device, registry, { ballCount: pairCount, lineSpanWidth, lineY, radius });
  const dropState = _initializeDropState(device, registry, { ballCount: pairCount, lineSpanWidth, lineY, radius });
  const pairState = _initializePairStateBuffers(device, registry, pairCount);
  return { anchorBuffer, dropState, pairState };
}