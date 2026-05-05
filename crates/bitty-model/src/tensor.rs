use bitty_protocol::Quantization;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShape(pub Vec<usize>);

impl TensorShape {
    pub fn element_count(&self) -> usize {
        self.0.iter().product()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LowBitTensor {
    pub shape: TensorShape,
    pub quantization: Quantization,
    pub packed_weights: Vec<u8>,
}

impl LowBitTensor {
    pub fn packed_len_for(elements: usize, quantization: Quantization) -> usize {
        match quantization {
            Quantization::F32 => elements * 4,
            Quantization::Fp16 => elements * 2,
            Quantization::Q8 => elements,
            Quantization::Q6 => (elements * 6).div_ceil(8),
            Quantization::Q5 => (elements * 5).div_ceil(8),
            Quantization::Q4 => elements.div_ceil(2),
            Quantization::Q3 => (elements * 3).div_ceil(8),
            Quantization::Q2 => elements.div_ceil(4),
            Quantization::Bit1 => elements.div_ceil(8),
        }
    }

    pub fn validate_len(&self) -> bool {
        self.packed_weights.len()
            == Self::packed_len_for(self.shape.element_count(), self.quantization)
    }
}
