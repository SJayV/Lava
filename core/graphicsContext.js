const MAX_DEVICE_PIXEL_RATIO = 2;

export async function initializeGraphicsContext(canvas) {
  if (!navigator.gpu) {
    throw new Error('graphicsContext: WebGPU is not available in this browser');
  }

  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const canvasContext = canvas.getContext('webgpu');
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  function _configure() {
    const pixelRatio = Math.min(window.devicePixelRatio, MAX_DEVICE_PIXEL_RATIO);
    canvas.width = Math.max(1, Math.floor(canvas.clientWidth * pixelRatio));
    canvas.height = Math.max(1, Math.floor(canvas.clientHeight * pixelRatio));
    canvasContext.configure({ device, format: presentationFormat, alphaMode: 'opaque' });
  }

  _configure();
  new ResizeObserver(_configure).observe(canvas);

  return { device, canvas, canvasContext, presentationFormat };
}
