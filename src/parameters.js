export const UNIFORM_BUFFER_SIZE = 10 * 16;

export function makeParameterStore(initialValues) {
  return {
    values: { ...initialValues },
  };
}

export function getParameterValue(store, key) {
  return store.values[key];
}