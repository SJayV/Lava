const YELLOW = [0.85, 0.45, 0.15];
const ORANGE = [0.8, 0.35, 0.07];
const RED = [0.7, 0.15, 0.05];

export function getColorChunk() {
  return /* wgsl */ `
    fn computeTemperatureColor(heatValue: f32) -> vec3<f32> {
      let yellow = vec3<f32>(${YELLOW.join(', ')});
      let orange = vec3<f32>(${ORANGE.join(', ')});
      let red = vec3<f32>(${RED.join(', ')});
      let lowMix = mix(yellow, orange, smoothstep(0.32, 0.5, heatValue));
      return mix(lowMix, red, smoothstep(0.5, 0.68, heatValue));
    }

    fn computeShadedColor(temperatureColor: vec3<f32>, normal: vec3<f32>, viewDirection: vec3<f32>) -> vec3<f32> {
      const EDGE_SHARPNESS: f32 = 1.2;
      let facing = clamp(dot(normal, viewDirection), 0.0, 1.0);
      let edgeFactor = pow(1.0 - facing, EDGE_SHARPNESS);
      let shade = mix(1.0, 0.0, edgeFactor);
      return temperatureColor * shade;
    }
  `;
}