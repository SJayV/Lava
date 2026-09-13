export function makeParameterStore(initialValues) {
  return {
    values: { ...initialValues },
  };
}

export function getParameterValue(store, key) {
  return store.values[key];
}