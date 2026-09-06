const HASH_PRIME_X = 374761393;
const HASH_PRIME_Y = 668265263;
const HASH_PRIME_Z = 2147483647;
const HASH_MIX_A = 1274126177;

export function hashLattice3D(x, y, z) {
  let h = (x * HASH_PRIME_X + y * HASH_PRIME_Y + z * HASH_PRIME_Z) | 0;
  h = Math.imul(h ^ (h >>> 13), HASH_MIX_A);
  h = h ^ (h >>> 16);
  return (h >>> 0) / 4294967295;
}

export function computeValueNoise3D(x, y, z) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const z0 = Math.floor(z);
  const tx = x - x0;
  const ty = y - y0;
  const tz = z - z0;

  const lerp = (a, b, t) => a + (b - a) * t;
  const corner = (dx, dy, dz) => hashLattice3D(x0 + dx, y0 + dy, z0 + dz);

  const x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), tx);
  const x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), tx);
  const x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), tx);
  const x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), tx);
  const y0v = lerp(x00, x10, ty);
  const y1v = lerp(x01, x11, ty);
  return lerp(y0v, y1v, tz);
}

export function computeTurbulence({ position, time, octaves, frequency = 1, speed = 1, gain = 0.5 }) {
  const [x, y, z] = position;
  const shiftedZ = z * frequency + time * speed;
  const shiftedX = x * frequency;
  const shiftedY = y * frequency;

  let total = 0;
  let weightSum = 0;
  for (let k = 0; k < octaves; k += 1) {
    const octaveScale = 2 ** k;
    const weight = gain ** k;
    total += weight * computeValueNoise3D(shiftedX * octaveScale, shiftedY * octaveScale, shiftedZ * octaveScale);
    weightSum += weight;
  }
  return total / weightSum;
}

// ───── WGSL CHUNK ─────

export function getTurbulenceNoiseShaderChunk() {
  return /* wgsl */ `
    fn hashLattice3D(x: i32, y: i32, z: i32) -> f32 {
      var h = bitcast<u32>(x * ${HASH_PRIME_X} + y * ${HASH_PRIME_Y} + z * ${HASH_PRIME_Z});
      h = (h ^ (h >> 13u)) * ${HASH_MIX_A}u;
      h = h ^ (h >> 16u);
      return f32(h) / 4294967295.0;
    }

    fn computeValueNoise3D(position: vec3<f32>) -> f32 {
      let base = floor(position);
      let t = position - base;
      let x0 = i32(base.x);
      let y0 = i32(base.y);
      let z0 = i32(base.z);

      let x00 = mix(hashLattice3D(x0, y0, z0), hashLattice3D(x0 + 1, y0, z0), t.x);
      let x10 = mix(hashLattice3D(x0, y0 + 1, z0), hashLattice3D(x0 + 1, y0 + 1, z0), t.x);
      let x01 = mix(hashLattice3D(x0, y0, z0 + 1), hashLattice3D(x0 + 1, y0, z0 + 1), t.x);
      let x11 = mix(hashLattice3D(x0, y0 + 1, z0 + 1), hashLattice3D(x0 + 1, y0 + 1, z0 + 1), t.x);
      let y0v = mix(x00, x10, t.y);
      let y1v = mix(x01, x11, t.y);
      return mix(y0v, y1v, t.z);
    }

    fn computeTurbulence(position: vec3<f32>, time: f32, octaves: u32, frequency: f32, speed: f32) -> f32 {
      let gain = 0.5;
      let shifted = vec3<f32>(position.x * frequency, position.y * frequency, position.z * frequency + time * speed);

      var total = 0.0;
      var weightSum = 0.0;
      var weight = 1.0;
      for (var k = 0u; k < octaves; k = k + 1u) {
        let octaveScale = f32(1u << k);
        total = total + weight * computeValueNoise3D(shifted * octaveScale);
        weightSum = weightSum + weight;
        weight = weight * gain;
      }
      return total / weightSum;
    }
  `;
}