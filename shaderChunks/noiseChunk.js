const HASH_PRIME_X = 374761393;
const HASH_PRIME_Y = 668265263;
const HASH_PRIME_Z = 2147483647;
const HASH_MIX_A = 1274126177;

export function getHashChunk() {
  return `
    fn hashLattice3D(x: i32, y: i32, z: i32) -> f32 {
      var h = bitcast<u32>(x * ${HASH_PRIME_X} + y * ${HASH_PRIME_Y} + z * ${HASH_PRIME_Z});
      h = (h ^ (h >> 13u)) * ${HASH_MIX_A}u;
      h = h ^ (h >> 16u);
      return f32(h) / 4294967295.0;
    }
  `;
}

export function getNoiseChunk() {
  return `
    ${getHashChunk()}

    fn hashGradient3D(x: i32, y: i32, z: i32) -> vec3<f32> {
      let theta = hashLattice3D(x, y, z) * 6.28318530718;
      let cosPhi = hashLattice3D(x, y, z + 1) * 2.0 - 1.0;
      let sinPhi = sqrt(max(0.0, 1.0 - cosPhi * cosPhi));
      return vec3<f32>(sinPhi * cos(theta), sinPhi * sin(theta), cosPhi);
    }

    struct LatticeCell {
      base: vec3<i32>,
      frac: vec3<f32>,
    }

    fn initializeLatticeCell(position: vec3<f32>) -> LatticeCell {
      let base = floor(position);
      return LatticeCell(vec3<i32>(base), position - base);
    }

    fn computeGradientNoise3D(position: vec3<f32>) -> f32 {
      let cell = initializeLatticeCell(position);
      let x0 = cell.base.x;
      let y0 = cell.base.y;
      let z0 = cell.base.z;
      let f = cell.frac;
      let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);

      let n000 = dot(hashGradient3D(x0, y0, z0), f - vec3<f32>(0.0, 0.0, 0.0));
      let n100 = dot(hashGradient3D(x0 + 1, y0, z0), f - vec3<f32>(1.0, 0.0, 0.0));
      let n010 = dot(hashGradient3D(x0, y0 + 1, z0), f - vec3<f32>(0.0, 1.0, 0.0));
      let n110 = dot(hashGradient3D(x0 + 1, y0 + 1, z0), f - vec3<f32>(1.0, 1.0, 0.0));
      let n001 = dot(hashGradient3D(x0, y0, z0 + 1), f - vec3<f32>(0.0, 0.0, 1.0));
      let n101 = dot(hashGradient3D(x0 + 1, y0, z0 + 1), f - vec3<f32>(1.0, 0.0, 1.0));
      let n011 = dot(hashGradient3D(x0, y0 + 1, z0 + 1), f - vec3<f32>(0.0, 1.0, 1.0));
      let n111 = dot(hashGradient3D(x0 + 1, y0 + 1, z0 + 1), f - vec3<f32>(1.0, 1.0, 1.0));

      let nx00 = mix(n000, n100, u.x);
      let nx10 = mix(n010, n110, u.x);
      let nx01 = mix(n001, n101, u.x);
      let nx11 = mix(n011, n111, u.x);
      let nxy0 = mix(nx00, nx10, u.y);
      let nxy1 = mix(nx01, nx11, u.y);
      return mix(nxy0, nxy1, u.z);
    }

    fn computeValueNoise3D(position: vec3<f32>) -> f32 {
      let cell = initializeLatticeCell(position);
      let x0 = cell.base.x;
      let y0 = cell.base.y;
      let z0 = cell.base.z;
      let t = cell.frac;

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
