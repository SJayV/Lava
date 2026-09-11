import { describe, it, expect } from 'vitest';
import { makeResourceRegistry, registerBuffer, getBuffer } from '../../src/gpuSetup.js';

function makeFakeDevice() {
  return {
    createBuffer: (descriptor) => ({ kind: 'buffer', descriptor }),
  };
}

describe('resourceRegistry', () => {
  it('starts with an empty buffer map', () => {
    const registry = makeResourceRegistry(makeFakeDevice());

    expect(registry.buffers.size).toBe(0);
  });

  it('creates and stores a buffer under the given name', () => {
    const registry = makeResourceRegistry(makeFakeDevice());

    const buffer = registerBuffer(registry, 'dropStateA', { size: 32 });

    expect(buffer.descriptor).toEqual({ size: 32 });
    expect(getBuffer(registry, 'dropStateA')).toBe(buffer);
  });

  it('throws when reading a buffer that was never registered', () => {
    const registry = makeResourceRegistry(makeFakeDevice());

    expect(() => getBuffer(registry, 'missing')).toThrow();
  });
});