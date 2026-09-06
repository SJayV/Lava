import { initializeGraphicsContext } from './core/graphicsContext.js';
import { makeResourceRegistry } from './core/resourceRegistry.js';
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
} from './simulation/dropState.js';
import { makeDropRaymarcher, writeRaymarchUniforms, renderRaymarchPass } from './rendering/dropRaymarcher.js';

const BALL_COUNT = 6;
const FIXED_TIMESTEP = 1 / 120;
const MAXIMUM_SUBSTEPS_PER_FRAME = 8;

const CAMERA_EYE = [0, -0.6, 4];
const CAMERA_TARGET = [0, -0.6, 0];
const CAMERA_UP = [0, 1, 0];
const FOV_VERTICAL = Math.PI / 4;

const FLUID_DENSITY = 1000;
const N_LOCAL = 4;
const TRACE_BOUND_MARGIN_IN_H = 2;
const RADIUS_TO_H_RATIO = 0.22;
const ISO_LEVEL_C = computeIsoLevelCalibrationFactorForRadiusRatio(RADIUS_TO_H_RATIO);

const LINE_Y = 1.2;

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

  const H = computeSmoothingRadiusFromLineSpan({ ballCount: BALL_COUNT, lineSpanWidth });
  const DROP_RADIUS = RADIUS_TO_H_RATIO * H;
  const MAX_RAY_DISTANCE = lineSpanWidth + TRACE_BOUND_MARGIN_IN_H * H;
  const TRACE_HALF_EXTENTS = [
    lineSpanWidth / 2 + TRACE_BOUND_MARGIN_IN_H * H,
    TRACE_BOUND_MARGIN_IN_H * H,
    TRACE_BOUND_MARGIN_IN_H * H,
  ];

  const seedPositions = computeLineSeedPositions({ ballCount: BALL_COUNT, lineSpanWidth, lineY: LINE_Y });
  const drops = seedPositions.map((position) => makeDropRecord({ position, radius: DROP_RADIUS }));
  const packedDrops = packDropRecords(drops);

  const dropState = makeDropState(graphicsContext.device, registry, BALL_COUNT);
  graphicsContext.device.queue.writeBuffer(getCurrentDropBuffer(dropState), 0, packedDrops);

  const raymarcher = makeDropRaymarcher(graphicsContext.device, graphicsContext.presentationFormat);

  const mass = computeParticleMass(DROP_RADIUS, FLUID_DENSITY);
  const isoLevel = computeIsoLevel(mass, H, ISO_LEVEL_C);
  const gradientMagnitudeMax = computeGradientMagnitudeBound(mass, H, N_LOCAL);

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
      dropCount: BALL_COUNT,
      minStep: 0.02 * H,
      maxStep: 0.5 * H,
      surfaceEpsilon: 0.0015,
      maxTraceSteps: 64,
      backgroundColor: [0.02, 0.02, 0.03],
      gradientMagnitudeMax,
    });

    const commandEncoder = graphicsContext.device.createCommandEncoder();
    renderRaymarchPass(
      raymarcher,
      commandEncoder,
      graphicsContext.canvasContext.getCurrentTexture().createView(),
      getCurrentDropBuffer(dropState),
    );
    graphicsContext.device.queue.submit([commandEncoder.finish()]);
  }

  const frameLoop = makeFrameLoop({
    fixedTimestep: FIXED_TIMESTEP,
    maximumSubstepsPerFrame: MAXIMUM_SUBSTEPS_PER_FRAME,
    getTimeScale: () => getParameterValue(parameterStore, 'simulationTimeScale'),
    updateSimulation: () => {},
    renderFrame,
  });
  frameLoop.start();
}

main();