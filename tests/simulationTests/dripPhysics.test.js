import { describe, it, expect } from 'vitest';
import {
  computeCohesionKernel,
  computeCohesionForce,
  computeSemiImplicitEulerStep,
} from '../../simulation/dripPhysics.js';

describe('computeCohesionKernel (Akinci)', () => {
  it('is continuous at r = 0.5h (the 2x-coefficient fix)', () => {
    const h = 0.3;
    const justBelow = computeCohesionKernel(0.5 * h - 1e-6, h);
    const justAbove = computeCohesionKernel(0.5 * h + 1e-6, h);

    expect(justBelow).toBeCloseTo(justAbove, 3);
  });

  it('is zero at r = h', () => {
    expect(computeCohesionKernel(0.3, 0.3)).toBeCloseTo(0);
  });

  it('is zero beyond h', () => {
    expect(computeCohesionKernel(0.31, 0.3)).toBe(0);
  });

  it('is negative (repulsive) near r = 0', () => {
    expect(computeCohesionKernel(0.01, 0.3)).toBeLessThan(0);
  });

  it('is positive (attractive) between 0.5h and h', () => {
    const h = 0.3;
    expect(computeCohesionKernel(0.75 * h, h)).toBeGreaterThan(0);
  });
});

describe('computeCohesionForce', () => {
  const h = 0.3;
  const gamma = 100;
  const massI = 1;
  const massJ = 1;

  it('returns zero for coincident positions (r -> 0 guard)', () => {
    const force = computeCohesionForce([1, 2, 3], [1, 2, 3], massI, massJ, gamma, h);

    expect(force).toEqual([0, 0, 0]);
  });

  it('pulls i toward j in the attractive zone (0.5h to h)', () => {
    const r = 0.75 * h;
    const force = computeCohesionForce([r, 0, 0], [0, 0, 0], massI, massJ, gamma, h);

    expect(force[0]).toBeLessThan(0);
  });

  it('pushes i away from j in the repulsive zone (below 0.5h)', () => {
    const r = 0.1 * h;
    const force = computeCohesionForce([r, 0, 0], [0, 0, 0], massI, massJ, gamma, h);

    expect(force[0]).toBeGreaterThan(0);
  });

  it('is exactly zero beyond h', () => {
    const force = computeCohesionForce([h + 0.01, 0, 0], [0, 0, 0], massI, massJ, gamma, h);

    expect(force).toEqual([0, 0, 0]);
  });
});

describe('computeSemiImplicitEulerStep', () => {
  it('advances position by velocity * dt with no acceleration or damping', () => {
    const { position } = computeSemiImplicitEulerStep({
      position: [0, 0, 0],
      velocity: [1, 0, 0],
      acceleration: [0, 0, 0],
      damping: 0,
      dt: 0.5,
    });

    expect(position).toEqual([0.5, 0, 0]);
  });

  it('applies acceleration to velocity before advancing position', () => {
    const { velocity, position } = computeSemiImplicitEulerStep({
      position: [0, 0, 0],
      velocity: [0, 0, 0],
      acceleration: [0, -10, 0],
      damping: 0,
      dt: 0.1,
    });

    expect(velocity[1]).toBeCloseTo(-1);
    expect(position[1]).toBeCloseTo(-0.1);
  });

  it('damping shrinks velocity multiplicatively before the acceleration term is added', () => {
    const { velocity } = computeSemiImplicitEulerStep({
      position: [0, 0, 0],
      velocity: [10, 0, 0],
      acceleration: [0, 0, 0],
      damping: 5,
      dt: 0.1,
    });

    expect(velocity[0]).toBeCloseTo(10 * (1 - 5 * 0.1));
  });
});
