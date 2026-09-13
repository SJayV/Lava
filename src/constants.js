export const LINE_Y = 1.2;

export const PAIR_COUNT = 6;
export const FLUID_DENSITY = 100;
export const RESPAWN_Y = LINE_Y - 6;

export const UNIFORM_BUFFER_SIZE = 10 * 16;

// ───── PARAMETER STORE ─────

export function initializeParameterStore(initialValues) {
  return {
    values: { ...initialValues },
  };
}

export function getParameterValue(store, key) {
  return store.values[key];
}