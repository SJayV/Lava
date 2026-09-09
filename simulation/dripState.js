import { computeDensityKernel, computeParticleMass, computeRadiusFromMass } from '../rendering/densityField.js';

export function makePairs(pairCount) {
  const pairs = [];
  for (let i = 0; i < pairCount; i += 1) {
    pairs.push({ anchorIndex: 2 * i, dripIndex: 2 * i + 1 });
  }
  return pairs;
}

export function computeCompensatedAnchorRadius({
  anchorBaseRadius,
  dripBaseRadius,
  separation,
  smoothingRadius,
  fluidDensity,
}) {
  const anchorBaseMass = computeParticleMass(anchorBaseRadius, fluidDensity);
  const dripBaseMass = computeParticleMass(dripBaseRadius, fluidDensity);
  const kernelRatio = computeDensityKernel(separation, smoothingRadius) / computeDensityKernel(0, smoothingRadius);
  const missingMass = dripBaseMass * (1 - kernelRatio);
  return computeRadiusFromMass(anchorBaseMass + missingMass, fluidDensity);
}