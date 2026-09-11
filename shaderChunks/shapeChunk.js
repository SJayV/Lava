export const POLY6_NORMALIZATION = 315 / (64 * Math.PI);

export function getDensityKernelChunk() {
  return /* wgsl */ `
    fn computeDensityKernel(distance: f32, smoothingRadius: f32) -> f32 {
      if (distance < 0.0 || distance >= smoothingRadius) {
        return 0.0;
      }
      let term = smoothingRadius * smoothingRadius - distance * distance;
      return (${POLY6_NORMALIZATION} / pow(smoothingRadius, 9.0)) * term * term * term;
    }
  `;
}

export function getParticleMassChunk() {
  return /* wgsl */ `
    fn computeParticleMass(radius: f32, fluidDensity: f32) -> f32 {
      return (4.0 / 3.0) * ${Math.PI} * radius * radius * radius * fluidDensity;
    }
  `;
}