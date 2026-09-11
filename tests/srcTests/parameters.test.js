import { describe, it, expect } from 'vitest';
import { makeParameterStore, getParameterValue } from '../../src/parameters.js';

describe('parameterStore', () => {
  it('seeds values from the initial values object', () => {
    const store = makeParameterStore({ gravity: 9.8, gamma: 120 });

    expect(getParameterValue(store, 'gravity')).toBe(9.8);
    expect(getParameterValue(store, 'gamma')).toBe(120);
  });

  it('does not alias the initial values object passed in', () => {
    const initialValues = { gravity: 9.8 };
    const store = makeParameterStore(initialValues);

    store.values.gravity = 5;

    expect(initialValues.gravity).toBe(9.8);
  });
});