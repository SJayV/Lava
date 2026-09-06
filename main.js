import { initializeGraphicsContext } from './core/graphicsContext.js';
import { makeResourceRegistry, getBuffer } from './core/resourceRegistry.js';
import { makeFrameLoop } from './core/frameLoop.js';
import { makeParameterStore, getParameterValue } from './parameters/parameterStore.js';
import { computeCameraBasisVectors, computeLineSpanWidth } from './rendering/cameraProjection.js';
import {
  computeParticleMass,
  computeIsoLevel,
  computeGradientMagnitudeBound,
  computeIsoLevelCalibrationFactorForRadiusRatio,
} from './rendering/densityField.js';
import {
  computeLineSeedPositions,
  computeSmoothingRadiusFromLineSpan,
  makeDropRecord,
  packDropRecords,
  makeDropState,
  getCurrentDropBuffer,
  swapDropState,
} from './simulation/dropState.js';
import {
  computeInitialPhases,
  makePairState,
  swapPairState,
} from './simulation/dripState.js';
import {
  makeDripComputePass,
  writeDripPhysicsUniforms,
  makeDripComputeBindGroup,
  runDripComputePass,
} from './simulation/dripComputePass.js';
import {
  makeDropRaymarcher,
  writeRaymarchUniforms,
  makeRaymarchBindGroup,
  renderRaymarchPass,
} from './rendering/dropRaymarcher.js';

const PAIR_COUNT = 6;
const DROP_COUNT = 2 * PAIR_COUNT;
const ACTIVE_PAIR_INDEX = 3;
const FIXED_TIMESTEP = 1 / 120;
const MAXIMUM_SUBSTEPS_PER_FRAME = 8;

const CAMERA_EYE = [0, -0.6, 4];
const CAMERA_TARGET = [0, -0.6, 0];
const CAMERA_UP = [0, 1, 0];
const FOV_VERTICAL = Math.PI / 4;

const FLUID_DENSITY = 100;
const N_LOCAL = 4;
const TRACE_BOUND_MARGIN_IN_H = 1;
const ANCHOR_RADIUS_TO_H_RATIO = 0.3;
const DRIP_RADIUS_TO_H_RATIO = 0.3;
const ISO_LEVEL_C = computeIsoLevelCalibrationFactorForRadiusRatio(DRIP_RADIUS_TO_H_RATIO);

const LINE_Y = 0.8;
const RESPAWN_Y = LINE_Y - 6;
const MU = 9;
const GAMMA = 0.2;

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
  const ANCHOR_RADIUS = ANCHOR_RADIUS_TO_H_RATIO * H;
  const DRIP_RADIUS = DRIP_RADIUS_TO_H_RATIO * H;
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

  const dropState = makeDropState(graphicsContext.device, registry, DROP_COUNT);
  graphicsContext.device.queue.writeBuffer(getCurrentDropBuffer(dropState), 0, packedDrops);

  const initialPhases = computeInitialPhases(PAIR_COUNT, ACTIVE_PAIR_INDEX);
  const pairState = makePairState(graphicsContext.device, registry, PAIR_COUNT, initialPhases);

  const raymarcher = makeDropRaymarcher(graphicsContext.device, graphicsContext.presentationFormat);
  const computePass = makeDripComputePass(graphicsContext.device);

  const dropStateA = getBuffer(registry, 'dropStateA');
  const dropStateB = getBuffer(registry, 'dropStateB');
  const pairStateA = getBuffer(registry, 'pairStateA');
  const pairStateB = getBuffer(registry, 'pairStateB');

  // one bind group per ping-pong direction, built once — never recreated per frame
  const computeBindGroupsByActiveIndex = [
    makeDripComputeBindGroup(computePass, dropStateA, pairStateA, dropStateB, pairStateB),
    makeDripComputeBindGroup(computePass, dropStateB, pairStateB, dropStateA, pairStateA),
  ];
  const raymarchBindGroupsByActiveIndex = [
    makeRaymarchBindGroup(raymarcher, dropStateA),
    makeRaymarchBindGroup(raymarcher, dropStateB),
  ];

  const anchorMass = computeParticleMass(ANCHOR_RADIUS, FLUID_DENSITY);
  const dripMass = computeParticleMass(DRIP_RADIUS, FLUID_DENSITY);
  const isoLevel = computeIsoLevel(dripMass, H, ISO_LEVEL_C);
  const gradientMagnitudeMax = computeGradientMagnitudeBound(anchorMass, H, N_LOCAL);

  let frameCommandEncoder = null;

  function updateSimulation(dt) {
    writeDripPhysicsUniforms(computePass, {
      h: H,
      gamma: GAMMA,
      mu: MU,
      dt,
      fluidDensity: FLUID_DENSITY,
      respawnY: RESPAWN_Y,
      baseRadius: DRIP_RADIUS,
      anchorBaseRadius: ANCHOR_RADIUS,
      dropCount: DROP_COUNT,
      pairCount: PAIR_COUNT,
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
      surfaceEpsilon: 0.0015,
      maxTraceSteps: 32,
      backgroundColor: [0.02, 0.02, 0.03],
      gradientMagnitudeMax,
    });

    frameCommandEncoder ??= graphicsContext.device.createCommandEncoder();
    renderRaymarchPass(
      raymarcher,
      frameCommandEncoder,
      graphicsContext.canvasContext.getCurrentTexture().createView(),
      raymarchBindGroupsByActiveIndex[dropState.activeIndex],
    );
    graphicsContext.device.queue.submit([frameCommandEncoder.finish()]);
    frameCommandEncoder = null;
  }

  const frameLoop = makeFrameLoop({
    fixedTimestep: FIXED_TIMESTEP,
    maximumSubstepsPerFrame: MAXIMUM_SUBSTEPS_PER_FRAME,
    getTimeScale: () => getParameterValue(parameterStore, 'simulationTimeScale'),
    updateSimulation,
    renderFrame,
  });
  frameLoop.start();
}

main();