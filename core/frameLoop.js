export function advanceFrameLoop(loopState, elapsedSeconds, options) {
  const { fixedTimestep, maximumSubstepsPerFrame, getTimeScale, updateSimulation, renderFrame } = options;

  loopState.accumulatedSeconds += elapsedSeconds * getTimeScale();

  let remainingSubsteps = maximumSubstepsPerFrame;
  while (loopState.accumulatedSeconds >= fixedTimestep && remainingSubsteps > 0) {
    updateSimulation(fixedTimestep);
    loopState.accumulatedSeconds -= fixedTimestep;
    remainingSubsteps -= 1;
  }

  renderFrame();
}

export function makeFrameLoop(options) {
  const loopState = { accumulatedSeconds: 0 };
  let lastTimestampMs = null;

  function _onAnimationFrame(timestampMs) {
    if (lastTimestampMs !== null) {
      const elapsedSeconds = (timestampMs - lastTimestampMs) / 1000;
      advanceFrameLoop(loopState, elapsedSeconds, options);
    }
    lastTimestampMs = timestampMs;
    requestAnimationFrame(_onAnimationFrame);
  }

  function start() {
    requestAnimationFrame(_onAnimationFrame);
  }

  return { start };
}
