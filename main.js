import { initializeGraphicsContext, makeResourceRegistry, getBuffer, MAIN_TEXTURE_FORMAT, makePostProcessor, resizePostProcessorIfNeeded, getMainTextureView, runPostProcessPass } from './src/gpuSetup.js';
import { makeParameterStore, getParameterValue } from './src/parameters.js';
import { PAIR_COUNT, DROP_COUNT, FLUID_DENSITY, RESPAWN_Y } from './src/constants.js';
import { CAMERA_EYE, CAMERA_TARGET, CAMERA_UP, FOV_VERTICAL, TRACE_BOUND_MARGIN_IN_H, computeCameraBasisVectors, computeLineSpanWidth, makeDropRaymarcher, writeRaymarchUniforms, makeRaymarchBindGroup, renderRaymarchPass } from './src/renderer.js';
import { N_LOCAL, ANCHOR_RADIUS_TO_H_RATIO, DRIP_RADIUS_TO_H_RATIO, ISO_LEVEL_C, computeParticleMass, computeIsoLevel, computeIsoConsistentRadius, computeGradientMagnitudeBound, makeDripComputePass, writeDripPhysicsUniforms, makeDripComputeBindGroup, runDripComputePass } from './src/simulation.js';
import { LINE_Y, computeSmoothingRadiusFromLineSpan, computeLineSeedPositions, makeDropRecord, packDropRecords, makeDropState, getCurrentDropBuffer, swapDropState, initializePairState, makePairState, swapPairState } from './src/state.js';

const ACTIVE_PAIR_INDEX = 3;

// ───── INITIALIZATION ─────

async function main() {
  const canvas = document.getElementById('canvas');
  const graphicsContext = await initializeGraphicsContext(canvas);
  const registry = makeResourceRegistry(graphicsContext.device);
  const parameterStore = makeParameterStore({ simulationTimeScale: 1 });

  const aspectRatio = canvas.clientWidth / canvas.clientHeight;
  const eyeDistance = Math.hypot(...CAMERA_EYE.map((v, i) => v - CAMERA_TARGET[i]));
  const lineSpanWidth = computeLineSpanWidth({ eyeDistance, fovVertical: FOV_VERTICAL, aspectRatio });
  const { rightAxis, trueUpAxis, forwardAxis } = computeCameraBasisVectors(CAMERA_EYE, CAMERA_TARGET, CAMERA_UP);
  const focalLength = 1 / Math.tan(FOV_VERTICAL / 2);

  const H = computeSmoothingRadiusFromLineSpan({ ballCount: PAIR_COUNT, lineSpanWidth });
  const DRIP_RADIUS = DRIP_RADIUS_TO_H_RATIO * H;
  const dripMass = computeParticleMass(DRIP_RADIUS, FLUID_DENSITY);
  const isoLevel = computeIsoLevel(dripMass, H, ISO_LEVEL_C);
  const ANCHOR_RADIUS = computeIsoConsistentRadius({
    isoLevel,
    radiusRatio: ANCHOR_RADIUS_TO_H_RATIO,
    smoothingRadius: H,
    fluidDensity: FLUID_DENSITY,
  });
  const MAX_RAY_DISTANCE = lineSpanWidth + TRACE_BOUND_MARGIN_IN_H * H;
  const TRACE_HALF_EXTENTS = [
    lineSpanWidth / 2 + TRACE_BOUND_MARGIN_IN_H * H,
    TRACE_BOUND_MARGIN_IN_H * H + Math.abs(RESPAWN_Y),
    TRACE_BOUND_MARGIN_IN_H * H,
  ];

  const seedPositions = computeLineSeedPositions({ ballCount: PAIR_COUNT, lineSpanWidth, lineY: LINE_Y });
  const drops = seedPositions.flatMap((position) => [
    makeDropRecord({ position, radius: ANCHOR_RADIUS }),
    makeDropRecord({ position, radius: DRIP_RADIUS }),
  ]);
  const packedDrops = packDropRecords(drops);

  const dropState = makeDropState(registry, DROP_COUNT);
  graphicsContext.device.queue.writeBuffer(getCurrentDropBuffer(dropState), 0, packedDrops);

  const initialPairStates = Array.from({ length: PAIR_COUNT }, (_, pairIndex) =>
    initializePairState({ tNow: 0, startGrowing: pairIndex === ACTIVE_PAIR_INDEX }));
  const pairState = makePairState(graphicsContext.device, registry, PAIR_COUNT, initialPairStates);

  const raymarcher = makeDropRaymarcher(graphicsContext.device, MAIN_TEXTURE_FORMAT);
  const postProcessor = makePostProcessor(graphicsContext.device, graphicsContext.presentationFormat);
  const computePass = makeDripComputePass(graphicsContext.device);

  const dropStateA = getBuffer(registry, 'dropStateA');
  const dropStateB = getBuffer(registry, 'dropStateB');
  const pairStateA = getBuffer(registry, 'pairStateA');
  const pairStateB = getBuffer(registry, 'pairStateB');

  const computeBindGroupsByActiveIndex = [
    makeDripComputeBindGroup(computePass, dropStateA, pairStateA, dropStateB, pairStateB),
    makeDripComputeBindGroup(computePass, dropStateB, pairStateB, dropStateA, pairStateA),
  ];
  const raymarchBindGroupsByActiveIndex = [
    makeRaymarchBindGroup(raymarcher, dropStateA),
    makeRaymarchBindGroup(raymarcher, dropStateB),
  ];

  const anchorMass = computeParticleMass(ANCHOR_RADIUS, FLUID_DENSITY);
  const gradientMagnitudeMax = computeGradientMagnitudeBound(anchorMass, H, N_LOCAL);

  let frameCommandEncoder = null;
  let simulationElapsedTime = 0;

  function updateSimulation(dt) {
    simulationElapsedTime += dt;
    writeDripPhysicsUniforms(computePass, {
      h: H,
      tNow: simulationElapsedTime,
      dt,
      fluidDensity: FLUID_DENSITY,
      respawnY: RESPAWN_Y,
      baseRadius: DRIP_RADIUS,
      anchorBaseRadius: ANCHOR_RADIUS,
      dropCount: DROP_COUNT,
    });

    frameCommandEncoder ??= graphicsContext.device.createCommandEncoder();
    runDripComputePass(computePass, frameCommandEncoder, computeBindGroupsByActiveIndex[dropState.activeIndex], DROP_COUNT);

    swapDropState(dropState);
    swapPairState(pairState);
  }

  function renderFrame() {
    writeRaymarchUniforms(raymarcher, {
      cameraRight: rightAxis,
      cameraUp: trueUpAxis,
      cameraForward: forwardAxis,
      cameraEye: CAMERA_EYE,
      width: canvas.width,
      height: canvas.height,
      aspectRatio,
      focalLength,
      traceHalfExtents: TRACE_HALF_EXTENTS,
      maxRayDistance: MAX_RAY_DISTANCE,
      h: H,
      isoLevel,
      fluidDensity: FLUID_DENSITY,
      dropCount: DROP_COUNT,
      minStep: 0.02 * H,
      maxStep: 0.5 * H,
      maxTraceSteps: 20,
      backgroundColor: [0.02, 0.02, 0.03],
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

  const FIXED_TIMESTEP = 1 / 120;
  const MAXIMUM_SUBSTEPS_PER_FRAME = 8;

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