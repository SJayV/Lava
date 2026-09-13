import { initializeGraphicsContext, initializeResourceRegistry, MAIN_TEXTURE_FORMAT, initializePostProcessor, resizePostProcessor, getMainTextureView, runPostProcessPass } from './src/gpuSetup.js';
import { SHADER_SOURCE as BLOOM_SHADER_SOURCE } from './shaders/bloomShader.js';
import { PAIR_COUNT, FLUID_DENSITY, RESPAWN_Y, LINE_Y, initializeParameterStore, getParameterValue } from './src/constants.js';
import { CAMERA_EYE, CAMERA_TARGET, CAMERA_UP, FOV_VERTICAL, TRACE_BOUND_MARGIN_IN_H, computeCameraBasisVectors, computeLineSpanWidth, initializeSceneRenderer, renderScene } from './src/renderer.js';
import { N_LOCAL, ANCHOR_RADIUS_TO_H_RATIO, DRIP_RADIUS_TO_H_RATIO, computeCalibration, initializeDripSimulation, stepDripSimulation } from './src/simulation.js';
import { computeSmoothingRadiusFromLineSpan, initializeSceneState } from './src/state.js';

// ───── CONSTANTS ─────

const FIXED_TIMESTEP = 1 / 120;
const MAXIMUM_SUBSTEPS_PER_FRAME = 8;

const MIN_STEP_TO_H_RATIO = 0.02;
const MAX_STEP_TO_H_RATIO = 0.5;
const BLACK = [0.02, 0.02, 0.03];

// ───── HELPER FUNCTIONS - WORLD SETUP ─────

function computeTraceBounds({ lineSpanWidth, h }) {
  return {
    maxRayDistance: lineSpanWidth + TRACE_BOUND_MARGIN_IN_H * h,
    traceHalfExtents: [
      lineSpanWidth / 2 + TRACE_BOUND_MARGIN_IN_H * h,
      TRACE_BOUND_MARGIN_IN_H * h + Math.abs(RESPAWN_Y),
      TRACE_BOUND_MARGIN_IN_H * h,
    ],
  };
}

async function initializeWorld(canvas, device) {
  const aspectRatio = canvas.clientWidth / canvas.clientHeight;
  const eyeDistance = Math.hypot(...CAMERA_EYE.map((v, i) => v - CAMERA_TARGET[i]));
  const lineSpanWidth = computeLineSpanWidth({ eyeDistance, fovVertical: FOV_VERTICAL, aspectRatio });
  const cameraBasis = computeCameraBasisVectors(CAMERA_EYE, CAMERA_TARGET, CAMERA_UP);
  const focalLength = 1 / Math.tan(FOV_VERTICAL / 2);
  const h = computeSmoothingRadiusFromLineSpan({ ballCount: PAIR_COUNT, lineSpanWidth });

  const { dripRadius, anchorRadius, isoLevel, gradientMagnitudeMax } = await computeCalibration(device, { smoothingRadius: h, dripRadiusRatio: DRIP_RADIUS_TO_H_RATIO, anchorRadiusRatio: ANCHOR_RADIUS_TO_H_RATIO, fluidDensity: FLUID_DENSITY, localNeighborCount: N_LOCAL });
  const traceBounds = computeTraceBounds({ lineSpanWidth, h });

  return { aspectRatio, lineSpanWidth, cameraBasis, focalLength, h, dripRadius, anchorRadius, isoLevel, gradientMagnitudeMax, traceBounds };
}

// ───── INITIALIZATION ─────

async function initialize() {
  const canvas = document.getElementById('canvas');
  const graphicsContext = await initializeGraphicsContext(canvas);
  const registry = initializeResourceRegistry(graphicsContext.device);
  const parameterStore = initializeParameterStore({ simulationTimeScale: 0.6 });
  const world = await initializeWorld(canvas, graphicsContext.device);
  const { anchorBuffer, dripState, pairState } = initializeSceneState(graphicsContext.device, registry, { pairCount: PAIR_COUNT, lineSpanWidth: world.lineSpanWidth, lineY: LINE_Y, anchorRadius: world.anchorRadius, dripRadius: world.dripRadius });
  const dripSimulation = initializeDripSimulation(graphicsContext.device, anchorBuffer, dripState, pairState, { h: world.h, fluidDensity: FLUID_DENSITY, respawnY: RESPAWN_Y, baseRadius: world.dripRadius, pairCount: PAIR_COUNT });
  const sceneRenderer = initializeSceneRenderer(graphicsContext.device, MAIN_TEXTURE_FORMAT, anchorBuffer, dripState, { cameraBasis: world.cameraBasis, aspectRatio: world.aspectRatio, focalLength: world.focalLength, traceHalfExtents: world.traceBounds.traceHalfExtents, maxRayDistance: world.traceBounds.maxRayDistance, h: world.h, isoLevel: world.isoLevel, fluidDensity: FLUID_DENSITY, pairCount: PAIR_COUNT, minStep: MIN_STEP_TO_H_RATIO * world.h, maxStep: MAX_STEP_TO_H_RATIO * world.h, maxTraceSteps: 20, backgroundColor: BLACK, gradientMagnitudeMax: world.gradientMagnitudeMax });
  const postProcessor = initializePostProcessor(graphicsContext.device, graphicsContext.presentationFormat, BLOOM_SHADER_SOURCE);

  let frameCommandEncoder = null;

  function updateSimulation(dt) {
    frameCommandEncoder ??= graphicsContext.device.createCommandEncoder();
    stepDripSimulation(dripSimulation, frameCommandEncoder, dt);
  }

  function renderFrame() {
    resizePostProcessor(postProcessor, canvas.width, canvas.height);

    frameCommandEncoder ??= graphicsContext.device.createCommandEncoder();
    renderScene(sceneRenderer, frameCommandEncoder, getMainTextureView(postProcessor), {
      width: canvas.width,
      height: canvas.height,
      animationTime: performance.now() / 1000,
    });
    runPostProcessPass(postProcessor, frameCommandEncoder, graphicsContext.canvasContext.getCurrentTexture().createView());
    graphicsContext.device.queue.submit([frameCommandEncoder.finish()]);
    frameCommandEncoder = null;
  }

  // ───── ANIMATION LOOP ─────

  let accumulatedSeconds = 0;
  let lastTimestampMs = null;

  function onAnimationFrame(timestampMs) {
    if (lastTimestampMs !== null) {
      const elapsedSeconds = (timestampMs - lastTimestampMs) / 1000;
      accumulatedSeconds += elapsedSeconds * getParameterValue(parameterStore, 'simulationTimeScale');

      let remainingSubsteps = MAXIMUM_SUBSTEPS_PER_FRAME;
      while (accumulatedSeconds >= FIXED_TIMESTEP && remainingSubsteps > 0) {
        updateSimulation(FIXED_TIMESTEP);
        accumulatedSeconds -= FIXED_TIMESTEP;
        remainingSubsteps -= 1;
      }

      renderFrame();
    }
    lastTimestampMs = timestampMs;
    requestAnimationFrame(onAnimationFrame);
  }

  requestAnimationFrame(onAnimationFrame);
}

initialize();