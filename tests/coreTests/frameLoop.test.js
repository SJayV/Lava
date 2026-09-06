import { describe, it, expect, vi } from 'vitest';
import { advanceFrameLoop } from '../../core/frameLoop.js';

function makeHarness({ fixedTimestep = 1 / 120, maximumSubstepsPerFrame = 8, timeScale = 1 } = {}) {
  const updateSimulation = vi.fn();
  const renderFrame = vi.fn();
  const loopState = { accumulatedSeconds: 0 };
  const options = {
    fixedTimestep,
    maximumSubstepsPerFrame,
    getTimeScale: () => timeScale,
    updateSimulation,
    renderFrame,
  };
  return { loopState, options, updateSimulation, renderFrame };
}

describe('advanceFrameLoop', () => {
  it('does not step the simulation when elapsed time is below one fixed timestep', () => {
    const { loopState, options, updateSimulation, renderFrame } = makeHarness();

    advanceFrameLoop(loopState, 1 / 240, options);

    expect(updateSimulation).not.toHaveBeenCalled();
    expect(renderFrame).toHaveBeenCalledTimes(1);
  });

  it('steps the simulation once when elapsed time matches exactly one fixed timestep', () => {
    const { loopState, options, updateSimulation } = makeHarness({ fixedTimestep: 1 / 120 });

    advanceFrameLoop(loopState, 1 / 120, options);

    expect(updateSimulation).toHaveBeenCalledTimes(1);
    expect(updateSimulation).toHaveBeenCalledWith(1 / 120);
  });

  it('drains multiple substeps when elapsed time covers several fixed timesteps', () => {
    const { loopState, options, updateSimulation } = makeHarness({ fixedTimestep: 1 / 120 });

    advanceFrameLoop(loopState, 3 / 120, options);

    expect(updateSimulation).toHaveBeenCalledTimes(3);
  });

  it('always passes the unscaled fixedTimestep to updateSimulation, never a time-scaled dt', () => {
    const { loopState, options, updateSimulation } = makeHarness({ fixedTimestep: 1 / 120, timeScale: 0.18 });

    // enough real elapsed time that, once scaled down, still covers one substep
    advanceFrameLoop(loopState, (1 / 120) / 0.18, options);

    expect(updateSimulation).toHaveBeenCalledWith(1 / 120);
  });

  it('scales accumulation by getTimeScale before draining substeps', () => {
    const { loopState, options, updateSimulation } = makeHarness({ fixedTimestep: 1 / 120, timeScale: 0.5 });

    // real elapsed time for one substep, but scaled by 0.5 it is not enough yet
    advanceFrameLoop(loopState, 1 / 120, options);

    expect(updateSimulation).not.toHaveBeenCalled();
  });

  it('caps substeps per frame and retains the leftover backlog for the next frame', () => {
    const { loopState, options, updateSimulation } = makeHarness({ fixedTimestep: 1 / 120, maximumSubstepsPerFrame: 2 });

    advanceFrameLoop(loopState, 5 / 120, options);

    expect(updateSimulation).toHaveBeenCalledTimes(2);
    expect(loopState.accumulatedSeconds).toBeCloseTo(3 / 120);
  });

  it('calls renderFrame exactly once per call regardless of how many substeps ran', () => {
    const { loopState, options, renderFrame } = makeHarness({ fixedTimestep: 1 / 120 });

    advanceFrameLoop(loopState, 10 / 120, options);

    expect(renderFrame).toHaveBeenCalledTimes(1);
  });
});
