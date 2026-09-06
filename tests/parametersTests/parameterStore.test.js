import { describe, it, expect, vi } from 'vitest';
import {
  makeParameterStore,
  getParameterValue,
  setParameterValue,
  subscribeToChanges,
} from '../../parameters/parameterStore.js';

describe('parameterStore', () => {
  it('seeds values from the initial values object', () => {
    const store = makeParameterStore({ gravity: 9.8, gamma: 120 });

    expect(getParameterValue(store, 'gravity')).toBe(9.8);
    expect(getParameterValue(store, 'gamma')).toBe(120);
  });

  it('does not alias the initial values object passed in', () => {
    const initialValues = { gravity: 9.8 };
    const store = makeParameterStore(initialValues);

    setParameterValue(store, 'gravity', 5);

    expect(initialValues.gravity).toBe(9.8);
  });

  it('reflects a value change immediately through getParameterValue', () => {
    const store = makeParameterStore({ gravity: 9.8 });

    setParameterValue(store, 'gravity', 5);

    expect(getParameterValue(store, 'gravity')).toBe(5);
  });

  it('notifies subscribers with the key and new value on change', () => {
    const store = makeParameterStore({ gravity: 9.8 });
    const listener = vi.fn();
    subscribeToChanges(store, listener);

    setParameterValue(store, 'gravity', 5);

    expect(listener).toHaveBeenCalledWith('gravity', 5);
  });

  it('stops notifying a subscriber once it unsubscribes', () => {
    const store = makeParameterStore({ gravity: 9.8 });
    const listener = vi.fn();
    const unsubscribe = subscribeToChanges(store, listener);

    unsubscribe();
    setParameterValue(store, 'gravity', 5);

    expect(listener).not.toHaveBeenCalled();
  });

  it('supports multiple independent subscribers', () => {
    const store = makeParameterStore({ gravity: 9.8 });
    const listenerA = vi.fn();
    const listenerB = vi.fn();
    subscribeToChanges(store, listenerA);
    subscribeToChanges(store, listenerB);

    setParameterValue(store, 'gravity', 5);

    expect(listenerA).toHaveBeenCalledWith('gravity', 5);
    expect(listenerB).toHaveBeenCalledWith('gravity', 5);
  });
});
