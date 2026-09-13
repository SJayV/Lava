import { initializeGraphicsContext, makeResourceRegistry, MAIN_TEXTURE_FORMAT, makePostProcessor, resizePostProcessorIfNeeded, getMainTextureView, runPostProcessPass } from './src/gpuSetup.js';
import { SHADER_SOURCE as BLOOM_SHADER_SOURCE } from './shaders/bloomShader.js';
import { makeParameterStore, getParameterValue } from './src/parameters.js';
import { PAIR_COUNT, DROP_COUNT, FLUID_DENSITY, RESPAWN_Y } from './src/constants.js';
import { CAMERA_EYE, CAMERA_TARGET, CAMERA_UP, FOV_VERTICAL, TRACE_BOUND_MARGIN_IN_H, computeCameraBasisVectors, computeLineSpanWidth, makeDropRaymarcher, writeRaymarchUniforms, makeRaymarchBindGroup, renderRaymarchPass } from './src/renderer.js';
import { N_LOCAL, ANCHOR_RADIUS_TO_H_RATIO, DRIP_RADIUS_TO_H_RATIO, computeCalibration, makeDripComputePass, writeDripPhysicsUniforms, makeDripComputeBindGroup, runDripComputePass } from './src/simulation.js';
import { LINE_Y, computeSmoothingRadiusFromLineSpan, computeLineSeedPositions, makeDropRecord, packDropRecords, makeDropState, getCurrentDropBuffer, getDropBufferPair, swapDropState, initializePairState, makePairState, getPairStateBufferPair, swapPairState } from './src/state.js';

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

// ───── HELPER FUNCTIONS - DROP/PAIR SEEDING ─────

function seedDropState(device, registry, { lineSpanWidth, dripRadius, anchorRadius }) {
  const seedPositions = computeLineSeedPositions({ ballCount: PAIR_COUNT, lineSpanWidth, lineY: LINE_Y });
  const drops = seedPositions.flatMap((position) => [
    makeDropRecord({ position, radius: anchorRadius }),
    makeDropRecord({ position, radius: dripRadius }),
  ]);

  const dropState = makeDropState(registry, DROP_COUNT);
  device.queue.writeBuffer(getCurrentDropBuffer(dropState), 0, packDropRecords(drops));
  return dropState;
}

function seedPairState(device, registry) {
  const initialPairStates = Array.from({ length: PAIR_COUNT }, () => initializePairState({ tNow: 0 }));
  return makePairState(device, registry, PAIR_COUNT, initialPairStates);
}

// ───── HELPER FUNCTIONS - BIND GROUPS ─────

function makeDripBindGroupsByActiveIndex(computePass, dropState, pairState) {
  const [dropStateA, dropStateB] = getDropBufferPair(dropState);
  const [pairStateA, pairStateB] = getPairStateBufferPair(pairState);
  return [
    makeDripComputeBindGroup(computePass, dropStateA, pairStateA, dropStateB, pairStateB),
    makeDripComputeBindGroup(computePass, dropStateB, pairStateB, dropStateA, pairStateA),
  ];
}

function makeRaymarchBindGroupsByActiveIndex(raymarcher, dropState) {
  const [dropStateA, dropStateB] = getDropBufferPair(dropState);
  return [
    makeRaymarchBindGroup(raymarcher, dropStateA),
    makeRaymarchBindGroup(raymarcher, dropStateB),
  ];
}

// ───── INITIALIZATION ─────

async function main() {
  const canvas = document.getElementById('canvas');
  const graphicsContext = await initializeGraphicsContext(canvas);
  const registry = makeResourceRegistry(graphicsContext.device);
  const parameterStore = makeParameterStore({ simulationTimeScale: 1 });

  const world = computeWorldSetup(canvas);
  const { dripRadius, anchorRadius, isoLevel, gradientMagnitudeMax } = await computeCalibration(graphicsContext.device, {
    smoothingRadius: world.h,
    dripRadiusRatio: DRIP_RADIUS_TO_H_RATIO,
    anchorRadiusRatio: ANCHOR_RADIUS_TO_H_RATIO,
    fluidDensity: FLUID_DENSITY,
    localNeighborCount: N_LOCAL,
  });
  const traceBounds = computeTraceBounds({ lineSpanWidth: world.lineSpanWidth, h: world.h });

  const dropState = seedDropState(graphicsContext.device, registry, { lineSpanWidth: world.lineSpanWidth, dripRadius, anchorRadius });
  const pairState = seedPairState(graphicsContext.device, registry);

  const raymarcher = makeDropRaymarcher(graphicsContext.device, MAIN_TEXTURE_FORMAT);
  const postProcessor = makePostProcessor(graphicsContext.device, graphicsContext.presentationFormat, BLOOM_SHADER_SOURCE);
  const computePass = makeDripComputePass(graphicsContext.device);

  const computeBindGroupsByActiveIndex = makeDripBindGroupsByActiveIndex(computePass, dropState, pairState);
  const raymarchBindGroupsByActiveIndex = makeRaymarchBindGroupsByActiveIndex(raymarcher, dropState);

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
      anchorBaseRadius: anchorRadius,
      dropCount: DROP_COUNT,
    });

    frameCommandEncoder ??= graphicsContext.device.createCommandEncoder();
    runDripComputePass(computePass, frameCommandEncoder, computeBindGroupsByActiveIndex[dropState.activeIndex], DROP_COUNT);

    swapDropState(dropState);
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
      dropCount: DROP_COUNT,
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
      raymarchBindGroupsByActiveIndex[dropState.activeIndex],
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