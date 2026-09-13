import { UNIFORM_BUFFER_SIZE } from './constants.js';

// ───── PUBLIC INTERFACE ─────

export function initializeUniformBuffer(device) {
  return device.createBuffer({
    size: UNIFORM_BUFFER_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

export function writeUniformBuffer(pass, data) {
  pass.device.queue.writeBuffer(pass.uniformBuffer, 0, data);
}

export function initializeBufferBindGroupLayout(device, visibility, bufferTypes) {
  return device.createBindGroupLayout({
    entries: bufferTypes.map((type, binding) => ({ binding, visibility, buffer: { type } })),
  });
}

export function initializeGpuPass(device, { visibility, bufferTypes, createPipeline }) {
  const uniformBuffer = initializeUniformBuffer(device);
  const bindGroupLayout = initializeBufferBindGroupLayout(device, visibility, bufferTypes);
  const pipeline = createPipeline(bindGroupLayout);
  return { device, uniformBuffer, bindGroupLayout, pipeline };
}

export function initializeBufferBindGroup(pass, buffers) {
  return pass.device.createBindGroup({
    layout: pass.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: pass.uniformBuffer } },
      ...buffers.map((buffer, index) => ({ binding: index + 1, resource: { buffer } })),
    ],
  });
}

export function dispatchComputePass(commandEncoder, pipeline, bindGroup, workgroupCount) {
  const pass = commandEncoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroupCount);
  pass.end();
}

export function drawFullscreenPass(commandEncoder, pipeline, bindGroup, targetView) {
  const pass = commandEncoder.beginRenderPass({
    colorAttachments: [{ view: targetView, loadOp: 'clear', storeOp: 'store' }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(3);
  pass.end();
}