use candle_core::Device;

pub fn auto_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            tracing::info!("selected CUDA device 0");
            return device;
        }
        tracing::debug!("CUDA device not available, falling back");
    }
    #[cfg(feature = "rocm")]
    {
        // TODO: implement when candle-core ships a ROCm backend.
        //       Currently Device::new_rocm does not exist in candle-core 0.10.x.
        tracing::info!("ROCm support requested but candle-core 0.10.x does not expose a ROCm device yet. Falling through to next backend.");
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            tracing::info!("selected Metal device 0");
            return device;
        }
        tracing::debug!("Metal device not available, falling back");
    }
    tracing::info!("falling back to CPU");
    Device::Cpu
}
