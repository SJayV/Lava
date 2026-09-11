import { describe, it, expect } from 'vitest';
import { computeCameraBasisVectors, computeLineSpanWidth } from '../../src/renderer.js';

describe('computeCameraBasisVectors', () => {
  it('produces an orthonormal right/up/forward basis for the default frontal camera', () => {
    const { rightAxis, trueUpAxis, forwardAxis } = computeCameraBasisVectors(
      [0, -0.6, 4],
      [0, -0.6, 0],
      [0, 1, 0],
    );

    expect(forwardAxis).toEqual([0, 0, 1]);
    expect(rightAxis).toEqual([1, 0, 0]);
    expect(trueUpAxis).toEqual([0, 1, 0]);
  });

  it('returns unit-length vectors', () => {
    const { rightAxis, trueUpAxis, forwardAxis } = computeCameraBasisVectors(
      [1, 2, 5],
      [0, -0.6, 0],
      [0, 1, 0],
    );

    for (const axis of [rightAxis, trueUpAxis, forwardAxis]) {
      const length = Math.hypot(axis[0], axis[1], axis[2]);
      expect(length).toBeCloseTo(1);
    }
  });

  it('keeps the three axes mutually perpendicular for a tilted eye position', () => {
    const { rightAxis, trueUpAxis, forwardAxis } = computeCameraBasisVectors(
      [2, 1, 3],
      [0, -0.6, 0],
      [0, 1, 0],
    );
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    expect(dot(rightAxis, trueUpAxis)).toBeCloseTo(0);
    expect(dot(trueUpAxis, forwardAxis)).toBeCloseTo(0);
    expect(dot(forwardAxis, rightAxis)).toBeCloseTo(0);
  });
});

describe('computeLineSpanWidth', () => {
  it('matches the analytic frustum-width formula at the target depth', () => {
    const eyeDistance = 4;
    const fovVertical = Math.PI / 4;
    const aspectRatio = 16 / 9;

    const width = computeLineSpanWidth({ eyeDistance, fovVertical, aspectRatio });

    const expected = 2 * eyeDistance * Math.tan(fovVertical / 2) * aspectRatio;
    expect(width).toBeCloseTo(expected);
  });

  it('grows with a wider aspect ratio', () => {
    const options = { eyeDistance: 4, fovVertical: Math.PI / 4 };

    const narrow = computeLineSpanWidth({ ...options, aspectRatio: 1.0 });
    const wide = computeLineSpanWidth({ ...options, aspectRatio: 2.0 });

    expect(wide).toBeGreaterThan(narrow);
  });
});