// ───── WGSL CHUNK ─────

export function getShadingShaderChunk() {
  return /* wgsl */ `
    fn computeShadedColor(temperatureColor: vec3<f32>, normal: vec3<f32>, viewDirection: vec3<f32>) -> vec3<f32> {
      const EDGE_DARKNESS: f32 = 0.0;
      const EDGE_SHARPNESS: f32 = 1.5;
      let facing = clamp(dot(normal, viewDirection), 0.0, 1.0);
      let edgeFactor = pow(1.0 - facing, EDGE_SHARPNESS);
      let shade = mix(1.0, EDGE_DARKNESS, edgeFactor);
      return temperatureColor * shade;
    }
  `;
}