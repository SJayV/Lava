const POLY6_NORMALIZATION = 315 / (64 * Math.PI);
const MASS_VOLUME_COEFFICIENT = (4 / 3) * Math.PI;

export function getDensityKernelChunk() {
  return `
    fn computeDensityKernel(distance: f32, smoothingRadius: f32) -> f32 {
      if (distance >= smoothingRadius) {
        return 0.0;
      }
      let term = smoothingRadius * smoothingRadius - distance * distance;
      return (${POLY6_NORMALIZATION} / pow(smoothingRadius, 9.0)) * term * term * term;
    }
  `;
}

export function getParticleMassChunk() {
  return `
    fn computeParticleMass(radius: f32, fluidDensity: f32) -> f32 {
      return ${MASS_VOLUME_COEFFICIENT} * radius * radius * radius * fluidDensity;
    }

    fn computeParticleRadius(mass: f32, fluidDensity: f32) -> f32 {
      return pow(mass / (${MASS_VOLUME_COEFFICIENT} * fluidDensity), 1.0 / 3.0);
    }
  `;
}