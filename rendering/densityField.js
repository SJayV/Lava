const POLY6_NORMALIZATION = 315 / (64 * Math.PI);
const POLY6_GRADIENT_MAX_COEFFICIENT = 2.7;

export function computeDensityKernel(distance, smoothingRadius) {
  if (distance < 0 || distance >= smoothingRadius) {
    return 0;
  }
  const term = smoothingRadius ** 2 - distance ** 2;
  return (POLY6_NORMALIZATION / smoothingRadius ** 9) * term ** 3;
}

export function computeParticleMass(radius, fluidDensity) {
  return (4 / 3) * Math.PI * radius ** 3 * fluidDensity;
}

export function computeRadiusFromMass(mass, fluidDensity) {
  return Math.cbrt(mass / ((4 / 3) * Math.PI * fluidDensity));
}

export function computeDensityField(point, drops, smoothingRadius) {
  let total = 0;
  for (const drop of drops) {
    const dx = point[0] - drop.position[0];
    const dy = point[1] - drop.position[1];
    const dz = point[2] - drop.position[2];
    const distance = Math.hypot(dx, dy, dz);
    total += drop.mass * computeDensityKernel(distance, smoothingRadius);
  }
  return total;
}

export function computeIsoLevel(mass, smoothingRadius, calibrationFactor) {
  return calibrationFactor * mass * computeDensityKernel(0, smoothingRadius);
}

export function computeIsosurfaceRadiusRatio(calibrationFactor) {
  return Math.sqrt(1 - calibrationFactor ** (1 / 3));
}

export function computeIsoLevelCalibrationFactorForRadiusRatio(radiusRatio) {
  return (1 - radiusRatio ** 2) ** 3;
}

export function computeIsoConsistentRadius({ isoLevel, radiusRatio, smoothingRadius, fluidDensity }) {
  const calibrationFactor = computeIsoLevelCalibrationFactorForRadiusRatio(radiusRatio);
  const requiredMass = isoLevel / (calibrationFactor * computeDensityKernel(0, smoothingRadius));
  return computeRadiusFromMass(requiredMass, fluidDensity);
}

export function computeGradientMagnitudeBound(mass, smoothingRadius, localNeighborCount) {
  const gradientKernelMax = POLY6_GRADIENT_MAX_COEFFICIENT / smoothingRadius ** 4;
  return localNeighborCount * mass * gradientKernelMax;
}

// ───── WGSL CHUNK ─────

export function getDensityFieldShaderChunk() {
  return /* wgsl */ `
    fn computeDensityKernel(distance: f32, smoothingRadius: f32) -> f32 {
      if (distance < 0.0 || distance >= smoothingRadius) {
        return 0.0;
      }
      let term = smoothingRadius * smoothingRadius - distance * distance;
      return (${POLY6_NORMALIZATION} / pow(smoothingRadius, 9.0)) * term * term * term;
    }

    fn computeParticleMass(radius: f32, fluidDensity: f32) -> f32 {
      return (4.0 / 3.0) * ${Math.PI} * radius * radius * radius * fluidDensity;
    }

    fn computeRadiusFromMass(mass: f32, fluidDensity: f32) -> f32 {
      return pow(mass / ((4.0 / 3.0) * ${Math.PI} * fluidDensity), 1.0 / 3.0);
    }
  `;
}