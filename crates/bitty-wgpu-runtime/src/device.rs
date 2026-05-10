use wgpu::{Backends, DeviceDescriptor, Features, Instance};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuBackend {
    Vulkan,
    Metal,
    Dx12,
    Auto,
}

impl GpuBackend {
    fn wgpu_backends(&self) -> Backends {
        match self {
            GpuBackend::Vulkan => Backends::VULKAN,
            GpuBackend::Metal => Backends::METAL,
            GpuBackend::Dx12 => Backends::DX12,
            GpuBackend::Auto => Backends::PRIMARY,
        }
    }
}

pub struct WgpuDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub backend: GpuBackend,
    pub adapter_info: String,
}

impl WgpuDevice {
    pub fn new(backend: GpuBackend) -> Result<Self, String> {
        let instance = Instance::new(&wgpu::InstanceDescriptor::default());
        let adapters: Vec<_> = instance.enumerate_adapters(Backends::all());
        let adapter = adapters
            .into_iter()
            .find(|a| {
                if backend == GpuBackend::Auto {
                    !matches!(a.get_info().device_type, wgpu::DeviceType::Cpu)
                } else {
                    true
                }
            })
            .ok_or_else(|| "No wgpu adapter found".to_string())?;

        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &DeviceDescriptor {
                required_features: Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: 512 * 1024 * 1024,
                    ..Default::default()
                },
                label: Some("bitty-wgpu"),
                memory_hints: Default::default(),
                trace: Default::default(),
            },
        ))
        .map_err(|e| format!("Failed to create wgpu device: {e}"))?;

        let backend_name = match info.backend {
            wgpu::Backend::Vulkan => GpuBackend::Vulkan,
            wgpu::Backend::Metal => GpuBackend::Metal,
            wgpu::Backend::Dx12 => GpuBackend::Dx12,
            _ => GpuBackend::Auto,
        };

        Ok(Self {
            device,
            queue,
            backend: backend_name,
            adapter_info: format!("{} ({:?})", info.name, info.backend),
        })
    }
}
