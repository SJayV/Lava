import { describe, it, expect } from 'vitest';
import {
  makeResourceRegistry,
  registerBuffer,
  registerTexture,
  getBuffer,
  getTextureIfRegistered,
} from '../../core/resourceRegistry.js';

// A GPUDevice stand-in: only the two factory calls the registry touches.
function makeFakeDevice() {
  return {
    createBuffer: (descriptor) => ({ kind: 'buffer', descriptor }),
    createTexture: (descriptor) => ({ kind: 'texture', descriptor }),
  };
}

describe('resourceRegistry', () => {
  it('starts with empty buffer and texture maps', () => {
    const registry = makeResourceRegistry(makeFakeDevice());

    expect(registry.buffers.size).toBe(0);
    expect(registry.textures.size).toBe(0);
  });

  it('creates and stores a buffer under the given name', () => {
    const registry = makeResourceRegistry(makeFakeDevice());

    const buffer = registerBuffer(registry, 'dropStateA', { size: 32 });

    expect(buffer.descriptor).toEqual({ size: 32 });
    expect(getBuffer(registry, 'dropStateA')).toBe(buffer);
  });

  it('creates and stores a texture under the given name', () => {
    const registry = makeResourceRegistry(makeFakeDevice());

    const texture = registerTexture(registry, 'raymarchColor', { width: 64 });

    expect(texture.descriptor).toEqual({ width: 64 });
    expect(getTextureIfRegistered(registry, 'raymarchColor')).toBe(texture);
  });

  it('throws when reading a buffer that was never registered', () => {
    const registry = makeResourceRegistry(makeFakeDevice());

    expect(() => getBuffer(registry, 'missing')).toThrow();
  });

  it('returns undefined for a texture that was never registered', () => {
    const registry = makeResourceRegistry(makeFakeDevice());

    expect(getTextureIfRegistered(registry, 'missing')).toBeUndefined();
  });
});
