import { initializeGraphicsContext } from './core/graphicsContext.js';
import { makeResourceRegistry } from './core/resourceRegistry.js';
import { makeFrameLoop } from './core/frameLoop.js';
import { makeParameterStore, getParameterValue } from './parameters/parameterStore.js';
import { computeCameraBasisVectors, computeLineSpanWidth } from './rendering/cameraProjection.js';
import { computeParticleMass } from './rendering/densityField.js';
import {
  computeLineSeedPositions,
  makeDropRecord,
  packDropRecords,
  makeDropState,
  getCurrentDropBuffer,
} from './simulation/dropState.js';
import { makeDropRaymarcher, writeRaymarchUniforms, renderRaymarchPass } from './rendering/dropRaymarcher.js';

const BALL_COUNT = 25;
const FIXED_TIMESTEP = 1 / 120;
const MAXIMUM_SUBSTEPS_PER_FRAME = 8;

const CAMERA_EYE = [0, -0.6, 4];
const CAMERA_TARGET = [0, -0.6, 0];
const CAMERA_UP = [0, 1, 0];
const FOV_VERTICAL = Math.PI / 4;
const MAX_RAY_DISTANCE = 20;

const H = 0.3;
const DROP_RADIUS = 0.3 * H;
const FLUID_DENSITY = 1000;
const ISO_LEVEL_C = 0.35;
const TRACE_HALF_EXTENTS = [8, 4, 4];

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

  const seedPositions = computeLineSeedPositions({ ballCount: BALL_COUNT, lineSpanWidth });
  const drops = seedPositions.map((position) => makeDropRecord({ position, radius: DROP_RADIUS }));
  const packedDrops = packDropRecords(drops);

  const dropState = makeDropState(graphicsContext.device, registry, BALL_COUNT);
  graphicsContext.device.queue.writeBuffer(getCurrentDropBuffer(dropState), 0, packedDrops);

  const raymarcher = makeDropRaymarcher(graphicsContext.device, graphicsContext.presentationFormat);

  const mass = computeParticleMass(DROP_RADIUS, FLUID_DENSITY);
  const isoLevel = ISO_LEVEL_C * mass * (315 / (64 * Math.PI * H ** 3));

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
