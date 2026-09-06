import { registerBuffer, getBuffer } from '../core/resourceRegistry.js';

const FLOATS_PER_DROP = 8;
const BYTES_PER_DROP = FLOATS_PER_DROP * 4;

export function computeLineSeedPositions({ ballCount, lineSpanWidth }) {
  const positions = [];
  for (let i = 0; i < ballCount; i += 1) {
    const x = -lineSpanWidth / 2 + (i / (ballCount - 1)) * lineSpanWidth;
    positions.push([x, 0, 0]);
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

// ───── GPU BUFFER WRAPPER ─────

export function makeDropState(device, registry, dropCount) {
  const size = dropCount * BYTES_PER_DROP;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  registerBuffer(registry, 'dropStateA', { size, usage });
  registerBuffer(registry, 'dropStateB', { size, usage });
  return { registry, dropCount, activeIndex: 0 };
}

export function getCurrentDropBuffer(dropState) {
  return getBuffer(dropState.registry, dropState.activeIndex === 0 ? 'dropStateA' : 'dropStateB');
}

export function getNextDropBuffer(dropState) {
  return getBuffer(dropState.registry, dropState.activeIndex === 0 ? 'dropStateB' : 'dropStateA');
}

export function swapDropState(dropState) {
  dropState.activeIndex = 1 - dropState.activeIndex;
}
