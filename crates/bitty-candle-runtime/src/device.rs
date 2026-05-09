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
