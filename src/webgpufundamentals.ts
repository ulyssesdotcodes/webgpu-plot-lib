async function fetchShader(device: GPUDevice, path: string, label: string): Promise<GPUShaderModule> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load shader ${path}: ${res.status}`);
  const code = await res.text();
  return device.createShaderModule({ label, code });
}


export async function init(canvas: HTMLCanvasElement) {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    const device = await adapter?.requestDevice();
    if (!device || !adapter) throw new Error('WebGPU not available');

    const context = canvas.getContext('webgpu');
    if (!context) throw new Error('canvas has no webgpu context');

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    const module = device.createShaderModule({
        label: 'our hardcoded rgb triangle shaders',
        code: await fetch("./shaders/webgpufundamentals/inter-stage-variables.wgsl").then(v => v.text()),
    });

    const pipeline = device.createRenderPipeline({
        label: 'hardcoded rgb triangle pipeline',
        layout: 'auto',
        vertex: {
        module,
        },
        fragment: {
        module,
        targets: [{ format }],
        },
    });


    function render() {
    // Get the current texture from the canvas context and
    // set it as the texture to render to.

    const renderPassDescriptor : GPURenderPassDescriptor = {
        label: 'our basic canvas renderPass',
        colorAttachments: [
        {
            // view: <- to be filled out when we render
            clearValue: [0.3, 0.3, 0.3, 1],
            loadOp: 'clear',
            storeOp: 'store',
            view: context!.getCurrentTexture().createView()
        },
        ],
    };

    const encoder = device!.createCommandEncoder({
      label: 'render triangle encoder',
    });
    const pass = encoder.beginRenderPass(renderPassDescriptor);
    pass.setPipeline(pipeline);
    pass.draw(3);  // call our vertex shader 3 times
    pass.end();

    const commandBuffer = encoder.finish();
    device!.queue.submit([commandBuffer]);
  }

  render();

}