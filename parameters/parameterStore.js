export function makeParameterStore(initialValues) {
  return {
    values: { ...initialValues },
    changeListeners: new Set(),
  };
}

export function getParameterValue(store, key) {
  return store.values[key];
}

export function setParameterValue(store, key, value) {
  store.values[key] = value;
  for (const listener of store.changeListeners) {
    listener(key, value);
  }
}

export function subscribeToChanges(store, listener) {
  store.changeListeners.add(listener);
  return () => store.changeListeners.delete(listener);
}