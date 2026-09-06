export function makeResourceRegistry(device) {
  return {
    device,
    textures: new Map(),
    buffers: new Map(),
  };
}

export function registerBuffer(registry, name, descriptor) {
  const buffer = registry.device.createBuffer(descriptor);
  registry.buffers.set(name, buffer);
  return buffer;
}

export function registerTexture(registry, name, descriptor) {
  const texture = registry.device.createTexture(descriptor);
  registry.textures.set(name, texture);
  return texture;
}

export function getBuffer(registry, name) {
  if (!registry.buffers.has(name)) {
    throw new Error(`resourceRegistry: no buffer registered under "${name}"`);
  }
  return registry.buffers.get(name);
}

export function getTextureIfRegistered(registry, name) {
  return registry.textures.get(name);
}
