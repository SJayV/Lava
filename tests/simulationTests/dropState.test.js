import { describe, it, expect } from 'vitest';
import { computeLineSeedPositions, makeDropRecord, packDropRecords } from '../../simulation/dropState.js';

describe('computeLineSeedPositions', () => {
  it('places the requested number of positions', () => {
    const positions = computeLineSeedPositions({ ballCount: 5, lineSpanWidth: 4 });

    expect(positions).toHaveLength(5);
  });

  it('evenly spans exactly [-lineSpanWidth/2, lineSpanWidth/2]', () => {
    const lineSpanWidth = 4;
    const positions = computeLineSeedPositions({ ballCount: 5, lineSpanWidth });

    expect(positions[0][0]).toBeCloseTo(-lineSpanWidth / 2);
    expect(positions[positions.length - 1][0]).toBeCloseTo(lineSpanWidth / 2);
  });

  it('spaces adjacent positions equally', () => {
    const positions = computeLineSeedPositions({ ballCount: 5, lineSpanWidth: 4 });

    const gaps = [];
    for (let i = 1; i < positions.length; i += 1) {
      gaps.push(positions[i][0] - positions[i - 1][0]);
    }
    for (const gap of gaps) {
      expect(gap).toBeCloseTo(gaps[0]);
    }
  });

  it('places every position on the y = 0, z = 0 line', () => {
    const positions = computeLineSeedPositions({ ballCount: 3, lineSpanWidth: 2 });

    for (const [, y, z] of positions) {
      expect(y).toBe(0);
      expect(z).toBe(0);
    }
  });
});

describe('makeDropRecord', () => {
  it('seeds a drop at rest: zero velocity, given position and radius', () => {
    const drop = makeDropRecord({ position: [1, 2, 3], radius: 0.1 });

    expect(drop.position).toEqual([1, 2, 3]);
    expect(drop.radius).toBe(0.1);
    expect(drop.velocity).toEqual([0, 0, 0]);
  });
});

describe('packDropRecords', () => {
  it('packs each drop into 8 floats: position, radius, velocity, speed', () => {
    const drops = [makeDropRecord({ position: [1, 2, 3], radius: 0.5 })];

    const packed = packDropRecords(drops);

    expect(packed).toBeInstanceOf(Float32Array);
    expect(packed).toHaveLength(8);
    expect(Array.from(packed)).toEqual([1, 2, 3, 0.5, 0, 0, 0, 0]);
  });

  it('packs multiple drops back to back in order', () => {
    const drops = [
      makeDropRecord({ position: [0, 0, 0], radius: 0.1 }),
      makeDropRecord({ position: [1, 0, 0], radius: 0.2 }),
    ];

    const packed = packDropRecords(drops);

    expect(packed).toHaveLength(16);
    // Float32Array rounds — compare within float32 precision, not exact equality.
    expect(packed[3]).toBeCloseTo(0.1);
    expect(packed[11]).toBeCloseTo(0.2);
  });

  it('caches the speed component as the magnitude of velocity', () => {
    const drop = { position: [0, 0, 0], radius: 0.1, velocity: [3, 4, 0] };

    const packed = packDropRecords([drop]);

    expect(packed[7]).toBeCloseTo(5);
  });
});
