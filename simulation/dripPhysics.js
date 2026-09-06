const COHESION_NORMALIZATION = 32 / Math.PI;

export function computeCohesionKernel(distance, smoothingRadius) {
  if (distance <= 0 || distance > smoothingRadius) {
    return 0;
  }
  const h6 = smoothingRadius ** 6;
  const nearTerm = (smoothingRadius - distance) ** 3 * distance ** 3;
  const value = distance <= 0.5 * smoothingRadius ? 2 * nearTerm - h6 / 64 : nearTerm;
  return (COHESION_NORMALIZATION / smoothingRadius ** 9) * value;
}

export function computeCohesionForce(positionI, positionJ, massI, massJ, gamma, smoothingRadius) {
  const offset = [positionI[0] - positionJ[0], positionI[1] - positionJ[1], positionI[2] - positionJ[2]];
  const distance = Math.hypot(offset[0], offset[1], offset[2]);
  if (distance < 1e-6 || distance > smoothingRadius) {
    return [0, 0, 0];
  }
  const kernel = computeCohesionKernel(distance, smoothingRadius);
  const scale = (-gamma * massI * massJ * kernel) / distance;
  return [offset[0] * scale, offset[1] * scale, offset[2] * scale];
}

export function computeSemiImplicitEulerStep({ position, velocity, acceleration, damping, dt }) {
  const nextVelocity = velocity.map((v, i) => v * (1 - damping * dt) + acceleration[i] * dt);
  const nextPosition = position.map((x, i) => x + nextVelocity[i] * dt);
  return { position: nextPosition, velocity: nextVelocity };
}

// ───── WGSL CHUNK ─────

export function getDripPhysicsShaderChunk() {
  return /* wgsl */ `
    fn computeCohesionKernel(distance: f32, smoothingRadius: f32) -> f32 {
      if (distance <= 0.0 || distance > smoothingRadius) {
        return 0.0;
      }
      let h6 = pow(smoothingRadius, 6.0);
      let nearTerm = pow(smoothingRadius - distance, 3.0) * pow(distance, 3.0);
      var value = nearTerm;
      if (distance <= 0.5 * smoothingRadius) {
        value = 2.0 * nearTerm - h6 / 64.0;
      }
      return (${COHESION_NORMALIZATION} / pow(smoothingRadius, 9.0)) * value;
    }

    fn computeCohesionForce(positionI: vec3<f32>, positionJ: vec3<f32>, massI: f32, massJ: f32, gamma: f32, smoothingRadius: f32) -> vec3<f32> {
      let offset = positionI - positionJ;
      let distance = length(offset);
      if (distance < 1e-6 || distance > smoothingRadius) {
        return vec3<f32>(0.0);
      }
      let kernel = computeCohesionKernel(distance, smoothingRadius);
      let scale = (-gamma * massI * massJ * kernel) / distance;
      return offset * scale;
    }
  `;
}