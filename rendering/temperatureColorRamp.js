export const YELLOW = [1.0, 0.8, 0.3];
export const ORANGE = [1.0, 0.55, 0.15];
export const RED = [0.9, 0.25, 0.1];

const _lerp = (a, b, t) => a + (b - a) * t;
const _smoothstep = (edge0, edge1, x) => {
  const t = Math.min(Math.max((x - edge0) / (edge1 - edge0), 0), 1);
  return t * t * (3 - 2 * t);
};
const _mixColor = (a, b, t) => [_lerp(a[0], b[0], t), _lerp(a[1], b[1], t), _lerp(a[2], b[2], t)];

export function computeTemperatureColor(heatValue) {
  const lowMix = _mixColor(YELLOW, ORANGE, _smoothstep(0.32, 0.5, heatValue));
  return _mixColor(lowMix, RED, _smoothstep(0.5, 0.68, heatValue));
}

// ───── WGSL CHUNK ─────

export function getTemperatureColorRampShaderChunk() {
  return /* wgsl */ `
    fn computeTemperatureColor(heatValue: f32) -> vec3<f32> {
      let yellow = vec3<f32>(${YELLOW.join(', ')});
      let orange = vec3<f32>(${ORANGE.join(', ')});
      let red = vec3<f32>(${RED.join(', ')});
      let lowMix = mix(yellow, orange, smoothstep(0.32, 0.5, heatValue));
      return mix(lowMix, red, smoothstep(0.5, 0.68, heatValue));
    }
  `;
}