import { initializeGraphicsContext, initializeResourceRegistry, MAIN_TEXTURE_FORMAT, initializePostProcessor, resizePostProcessor, getMainTextureView, runPostProcessPass } from './src/gpuSetup.js';
import { SHADER_SOURCE as BLOOM_SHADER_SOURCE } from './shaders/bloomShader.js';
import { PAIR_COUNT, FLUID_DENSITY, RESPAWN_Y, LINE_Y, initializeParameterStore, getParameterValue } from './src/constants.js';
import { CAMERA_EYE, CAMERA_TARGET, CAMERA_UP, FOV_VERTICAL, TRACE_BOUND_MARGIN_IN_H, computeCameraBasisVectors, computeDistance, computeLineSpanWidth, initializeSceneRenderer, renderScene } from './src/renderer.js';
import { N_LOCAL, ANCHOR_RADIUS_TO_H_RATIO, DRIP_RADIUS_TO_H_RATIO, computeCalibration, initializeDripSimulation, stepDripSimulation } from './src/simulation.js';
import { computeSmoothingRadiusFromLineSpan, initializeSceneState } from './src/state.js';

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

function computeAspectRatio(canvas) {
  return canvas.clientWidth / canvas.clientHeight;
}

function computeFocalLength(fovVertical) {
  return 1 / Math.tan(fovVertical / 2);
}

async function initializeWorld(canvas, device) {
  const aspectRatio = computeAspectRatio(canvas);
  const eyeDistance = computeDistance(CAMERA_EYE, CAMERA_TARGET);
  const lineSpanWidth = computeLineSpanWidth({ eyeDistance, fovVertical: FOV_VERTICAL, aspectRatio });
  const cameraBasis = computeCameraBasisVectors(CAMERA_EYE, CAMERA_TARGET, CAMERA_UP);
  const focalLength = computeFocalLength(FOV_VERTICAL);
  const h = computeSmoothingRadiusFromLineSpan({ ballCount: PAIR_COUNT, lineSpanWidth });

  const { dripRadius, anchorRadius, isoLevel, gradientMagnitudeMax } = await computeCalibration(device, { smoothingRadius: h, dripRadiusRatio: DRIP_RADIUS_TO_H_RATIO, anchorRadiusRatio: ANCHOR_RADIUS_TO_H_RATIO, fluidDensity: FLUID_DENSITY, localNeighborCount: N_LOCAL });
  const traceBounds = computeTraceBounds({ lineSpanWidth, h });

  return { aspectRatio, lineSpanWidth, cameraBasis, focalLength, h, dripRadius, anchorRadius, isoLevel, gradientMagnitudeMax, traceBounds };
}

// ───── HELPER FUNCTIONS - ANIMATION LOOP ─────

function advanceSimulation(accumulatedSeconds, step) {
  const FIXED_TIMESTEP = 1 / 120;
  let remainingSubsteps = 8;
  while (accumulatedSeconds >= FIXED_TIMESTEP && remainingSubsteps > 0) {
    step(FIXED_TIMESTEP);
    accumulatedSeconds -= FIXED_TIMESTEP;
    remainingSubsteps -= 1;
  }
  return accumulatedSeconds;
}

// ───── INITIALIZATION ─────

async function initialize() {
  const MIN_STEP_TO_H_RATIO = 0.02;
  const MAX_STEP_TO_H_RATIO = 0.5;

  const canvas = document.getElementById('canvas');
  const graphicsContext = await initializeGraphicsContext(canvas);
  const registry = initializeResourceRegistry(graphicsContext.device);
  const parameterStore = initializeParameterStore({ simulationTimeScale: 0.6 });
  const world = await initializeWorld(canvas, graphicsContext.device);
  const { anchorBuffer, dropState, pairState } = initializeSceneState(graphicsContext.device, registry, { pairCount: PAIR_COUNT, lineSpanWidth: world.lineSpanWidth, lineY: LINE_Y, anchorRadius: world.anchorRadius, dripRadius: world.dripRadius });
  const dripSimulation = initializeDripSimulation(graphicsContext.device, anchorBuffer, dropState, pairState, { h: world.h, fluidDensity: FLUID_DENSITY, respawnY: RESPAWN_Y, baseRadius: world.dripRadius, pairCount: PAIR_COUNT });
  const sceneRenderer = initializeSceneRenderer(graphicsContext.device, MAIN_TEXTURE_FORMAT, anchorBuffer, dropState, { cameraBasis: world.cameraBasis, aspectRatio: world.aspectRatio, focalLength: world.focalLength, traceHalfExtents: world.traceBounds.traceHalfExtents, maxRayDistance: world.traceBounds.maxRayDistance, h: world.h, isoLevel: world.isoLevel, fluidDensity: FLUID_DENSITY, pairCount: PAIR_COUNT, minStep: MIN_STEP_TO_H_RATIO * world.h, maxStep: MAX_STEP_TO_H_RATIO * world.h, maxTraceSteps: 20, backgroundColor: [0.02, 0.02, 0.03], gradientMagnitudeMax: world.gradientMagnitudeMax });
  const postProcessor = initializePostProcessor(graphicsContext.device, graphicsContext.presentationFormat, BLOOM_SHADER_SOURCE);

  let frameCommandEncoder = null;

  function getFrameCommandEncoder() {
    frameCommandEncoder ??= graphicsContext.device.createCommandEncoder();
    return frameCommandEncoder;
  }

  function updateSimulation(dt) {
    stepDripSimulation(dripSimulation, getFrameCommandEncoder(), dt);
  }

  function renderFrame() {
    resizePostProcessor(postProcessor, canvas.width, canvas.height);

    const encoder = getFrameCommandEncoder();
    renderScene(sceneRenderer, encoder, getMainTextureView(postProcessor), {
      width: canvas.width,
      height: canvas.height,
      animationTime: performance.now() / 1000,
    });
    runPostProcessPass(postProcessor, encoder, graphicsContext.canvasContext.getCurrentTexture().createView());
    graphicsContext.device.queue.submit([encoder.finish()]);
    frameCommandEncoder = null;
  }

  // ───── ANIMATION LOOP ─────

  let accumulatedSeconds = 0;
  let lastTimestampMs = null;

  function onAnimationFrame(timestampMs) {
    if (lastTimestampMs !== null) {
      const elapsedSeconds = (timestampMs - lastTimestampMs) / 1000;
      accumulatedSeconds += elapsedSeconds * getParameterValue(parameterStore, 'simulationTimeScale');
      accumulatedSeconds = advanceSimulation(accumulatedSeconds, updateSimulation);
      renderFrame();
    }
    lastTimestampMs = timestampMs;
    requestAnimationFrame(onAnimationFrame);
  }
  requestAnimationFrame(onAnimationFrame);
}

initialize();