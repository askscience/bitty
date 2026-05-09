use candle_core::Tensor;

#[derive(Clone)]
pub struct KvCache {
    pub seq_len: usize,
    pub cache_k: Option<Tensor>,
    pub cache_v: Option<Tensor>,
    max_seq_len: usize,
}

impl KvCache {
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            seq_len: 0,
            cache_k: None,
            cache_v: None,
            max_seq_len,
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> candle_core::Result<()> {
        let new_len = self.seq_len + k.dim(2)?;
        if new_len > self.max_seq_len {
            let excess = new_len - self.max_seq_len;
            self.cache_k = self.cache_k.as_ref().map(|ck| {
                ck.narrow(2, excess, ck.dim(2).unwrap_or(0) - excess).unwrap_or_else(|_| ck.clone())
            });
            self.cache_v = self.cache_v.as_ref().map(|cv| {
                cv.narrow(2, excess, cv.dim(2).unwrap_or(0) - excess).unwrap_or_else(|_| cv.clone())
            });
            self.seq_len -= excess;
        }

        self.cache_k = Some(match self.cache_k.take() {
            Some(ck) => Tensor::cat(&[&ck, k], 2)?,
            None => k.clone(),
        });
        self.cache_v = Some(match self.cache_v.take() {
            Some(cv) => Tensor::cat(&[&cv, v], 2)?,
            None => v.clone(),
        });
        self.seq_len = new_len.min(self.max_seq_len);
        Ok(())
    }

    pub fn reset(&mut self) {
        self.seq_len = 0;
        self.cache_k = None;
        self.cache_v = None;
    }
}
