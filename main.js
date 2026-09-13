import { initializeGraphicsContext, makeResourceRegistry, MAIN_TEXTURE_FORMAT, makePostProcessor, resizePostProcessorIfNeeded, getMainTextureView, runPostProcessPass } from './src/gpuSetup.js';
import { SHADER_SOURCE as BLOOM_SHADER_SOURCE } from './shaders/bloomShader.js';
import { makeParameterStore, getParameterValue } from './src/parameters.js';
import { PAIR_COUNT, FLUID_DENSITY, RESPAWN_Y, LINE_Y } from './src/constants.js';
import { CAMERA_EYE, CAMERA_TARGET, CAMERA_UP, FOV_VERTICAL, TRACE_BOUND_MARGIN_IN_H, computeCameraBasisVectors, computeLineSpanWidth, makeDropRaymarcher, writeRaymarchUniforms, makeRaymarchBindGroup, renderRaymarchPass } from './src/renderer.js';
import { N_LOCAL, ANCHOR_RADIUS_TO_H_RATIO, DRIP_RADIUS_TO_H_RATIO, computeCalibration, makeDripComputePass, writeDripPhysicsUniforms, makeDripComputeBindGroup, runDripComputePass } from './src/simulation.js';
import { computeSmoothingRadiusFromLineSpan, computeLineSeedPositions, makeDropRecord, packDropRecords, makeAnchorState, getAnchorBuffer, makeDripState, getCurrentDripBuffer, getDripBufferPair, swapDripState, initializePairState, makePairState, getPairStateBufferPair, swapPairState } from './src/state.js';

// ───── CONSTANTS ─────

const FIXED_TIMESTEP = 1 / 120;
const MAXIMUM_SUBSTEPS_PER_FRAME = 8;

const MIN_STEP_TO_H_RATIO = 0.02;
const MAX_STEP_TO_H_RATIO = 0.5;
const BLACK = [0.02, 0.02, 0.03];

// ───── HELPER FUNCTIONS - WORLD SETUP ─────

function computeWorldSetup(canvas) {
  const aspectRatio = canvas.clientWidth / canvas.clientHeight;
  const eyeDistance = Math.hypot(...CAMERA_EYE.map((v, i) => v - CAMERA_TARGET[i]));
  const lineSpanWidth = computeLineSpanWidth({ eyeDistance, fovVertical: FOV_VERTICAL, aspectRatio });
  const cameraBasis = computeCameraBasisVectors(CAMERA_EYE, CAMERA_TARGET, CAMERA_UP);
  const focalLength = 1 / Math.tan(FOV_VERTICAL / 2);
  const h = computeSmoothingRadiusFromLineSpan({ ballCount: PAIR_COUNT, lineSpanWidth });
  return { aspectRatio, lineSpanWidth, cameraBasis, focalLength, h };
}

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

// ───── HELPER FUNCTIONS - SEEDING ─────

function seedAnchorState(device, registry, { lineSpanWidth, anchorRadius }) {
  const seedPositions = computeLineSeedPositions({ ballCount: PAIR_COUNT, lineSpanWidth, lineY: LINE_Y });
  const anchorRecords = seedPositions.map((position) => makeDropRecord({ position, radius: anchorRadius }));
  return makeAnchorState(device, registry, PAIR_COUNT, anchorRecords);
}

function seedDripState(device, registry, { lineSpanWidth, dripRadius }) {
  const seedPositions = computeLineSeedPositions({ ballCount: PAIR_COUNT, lineSpanWidth, lineY: LINE_Y });
  const dripRecords = seedPositions.map((position) => makeDropRecord({ position, radius: dripRadius }));

  const dripState = makeDripState(registry, PAIR_COUNT);
  device.queue.writeBuffer(getCurrentDripBuffer(dripState), 0, packDropRecords(dripRecords));
  return dripState;
}

function seedPairState(device, registry) {
  const initialPairStates = Array.from({ length: PAIR_COUNT }, () => initializePairState({ tNow: Math.random() * 10 }));
  return makePairState(device, registry, PAIR_COUNT, initialPairStates);
}

// ───── HELPER FUNCTIONS - BIND GROUPS ─────

function makeDripBindGroupsByActiveIndex(computePass, anchorBuffer, dripState, pairState) {
  const [dripStateA, dripStateB] = getDripBufferPair(dripState);
  const [pairStateA, pairStateB] = getPairStateBufferPair(pairState);
  return [
    makeDripComputeBindGroup(computePass, anchorBuffer, dripStateA, pairStateA, dripStateB, pairStateB),
    makeDripComputeBindGroup(computePass, anchorBuffer, dripStateB, pairStateB, dripStateA, pairStateA),
  ];
}

function makeRaymarchBindGroupsByActiveIndex(raymarcher, anchorBuffer, dripState) {
  const [dripStateA, dripStateB] = getDripBufferPair(dripState);
  return [
    makeRaymarchBindGroup(raymarcher, anchorBuffer, dripStateA),
    makeRaymarchBindGroup(raymarcher, anchorBuffer, dripStateB),
  ];
}

// ───── INITIALIZATION ─────

async function main() {
  const canvas = document.getElementById('canvas');
  const graphicsContext = await initializeGraphicsContext(canvas);
  const registry = makeResourceRegistry(graphicsContext.device);
  const parameterStore = makeParameterStore({ simulationTimeScale: 0.6 });

  const world = computeWorldSetup(canvas);
  const { dripRadius, anchorRadius, isoLevel, gradientMagnitudeMax } = await computeCalibration(graphicsContext.device, {
    smoothingRadius: world.h,
    dripRadiusRatio: DRIP_RADIUS_TO_H_RATIO,
    anchorRadiusRatio: ANCHOR_RADIUS_TO_H_RATIO,
    fluidDensity: FLUID_DENSITY,
    localNeighborCount: N_LOCAL,
  });
  const traceBounds = computeTraceBounds({ lineSpanWidth: world.lineSpanWidth, h: world.h });

  const anchorState = seedAnchorState(graphicsContext.device, registry, { lineSpanWidth: world.lineSpanWidth, anchorRadius });
  const dripState = seedDripState(graphicsContext.device, registry, { lineSpanWidth: world.lineSpanWidth, dripRadius });
  const pairState = seedPairState(graphicsContext.device, registry);
  const anchorBuffer = getAnchorBuffer(anchorState);

  const raymarcher = makeDropRaymarcher(graphicsContext.device, MAIN_TEXTURE_FORMAT);
  const postProcessor = makePostProcessor(graphicsContext.device, graphicsContext.presentationFormat, BLOOM_SHADER_SOURCE);
  const computePass = makeDripComputePass(graphicsContext.device);

  const computeBindGroupsByActiveIndex = makeDripBindGroupsByActiveIndex(computePass, anchorBuffer, dripState, pairState);
  const raymarchBindGroupsByActiveIndex = makeRaymarchBindGroupsByActiveIndex(raymarcher, anchorBuffer, dripState);

  let frameCommandEncoder = null;
  let simulationElapsedTime = 0;

  function updateSimulation(dt) {
    simulationElapsedTime += dt;
    writeDripPhysicsUniforms(computePass, {
      h: world.h,
      tNow: simulationElapsedTime,
      dt,
      fluidDensity: FLUID_DENSITY,
      respawnY: RESPAWN_Y,
      baseRadius: dripRadius,
      pairCount: PAIR_COUNT,
    });

    frameCommandEncoder ??= graphicsContext.device.createCommandEncoder();
    runDripComputePass(computePass, frameCommandEncoder, computeBindGroupsByActiveIndex[dripState.activeIndex], PAIR_COUNT);

    swapDripState(dripState);
    swapPairState(pairState);
  }

  function renderFrame() {
    writeRaymarchUniforms(raymarcher, {
      cameraRight: world.cameraBasis.rightAxis,
      cameraUp: world.cameraBasis.trueUpAxis,
      cameraForward: world.cameraBasis.forwardAxis,
      cameraEye: CAMERA_EYE,
      width: canvas.width,
      height: canvas.height,
      aspectRatio: world.aspectRatio,
      focalLength: world.focalLength,
      traceHalfExtents: traceBounds.traceHalfExtents,
      maxRayDistance: traceBounds.maxRayDistance,
      h: world.h,
      isoLevel,
      fluidDensity: FLUID_DENSITY,
      pairCount: PAIR_COUNT,
      minStep: MIN_STEP_TO_H_RATIO * world.h,
      maxStep: MAX_STEP_TO_H_RATIO * world.h,
      maxTraceSteps: 20,
      backgroundColor: BLACK,
      gradientMagnitudeMax,
      animationTime: performance.now() / 1000,
    });

    resizePostProcessorIfNeeded(postProcessor, canvas.width, canvas.height);

    frameCommandEncoder ??= graphicsContext.device.createCommandEncoder();
    renderRaymarchPass(
      raymarcher,
      frameCommandEncoder,
      getMainTextureView(postProcessor),
      raymarchBindGroupsByActiveIndex[dripState.activeIndex],
    );
    runPostProcessPass(
      postProcessor,
      frameCommandEncoder,
      graphicsContext.canvasContext.getCurrentTexture().createView(),
    );
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

main();