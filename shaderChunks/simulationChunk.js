const COHESION_NORMALIZATION = 32 / Math.PI;

export function getSimulationChunk() {
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