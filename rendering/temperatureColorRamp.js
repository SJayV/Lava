export const YELLOW = [0.85, 0.45, 0.15];
export const ORANGE = [0.8, 0.35, 0.07];
export const RED = [0.7, 0.15, 0.05];

// ───── WGSL CHUNK ─────

export function getTemperatureColorRampShaderChunk() {
  return `
    fn computeTemperatureColor(heatValue: f32) -> vec3<f32> {
      let yellow = vec3<f32>(${YELLOW.join(', ')});
      let orange = vec3<f32>(${ORANGE.join(', ')});
      let red = vec3<f32>(${RED.join(', ')});
      let lowMix = mix(yellow, orange, smoothstep(0.32, 0.5, heatValue));
      return mix(lowMix, red, smoothstep(0.5, 0.68, heatValue));
    }
  `;
}