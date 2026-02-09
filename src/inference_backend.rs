// Inference-specific backend configuration for AlphaZero
// This file provides a clean separation between training and inference

use crate::alphazero::AlphaZeroNet;
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

// GPU backends (native burn-cuda via CubeCL)
#[cfg(feature = "cuda")]
pub type InferenceBackend = burn_cuda::Cuda;
#[cfg(feature = "cuda")]
pub type InferenceDevice = burn_cuda::CudaDevice;

#[cfg(not(feature = "cuda"))]
pub type InferenceBackend = burn_ndarray::NdArray;
#[cfg(not(feature = "cuda"))]
pub type InferenceDevice = burn_ndarray::NdArrayDevice;

// Training backends (with Autodiff wrapper)
#[cfg(feature = "cuda")]
pub type TrainingBackend = burn::backend::Autodiff<burn_cuda::Cuda>;

#[cfg(not(feature = "cuda"))]
pub type TrainingBackend = burn::backend::Autodiff<burn_ndarray::NdArray>;

/// Load a trained model for inference
pub fn load_model_for_inference(
    model_path: &str,
    device: &InferenceDevice,
) -> Result<AlphaZeroNet<InferenceBackend>, Box<dyn std::error::Error>> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    match recorder.load(model_path.into(), device) {
        Ok(record) => {
            let model = AlphaZeroNet::<InferenceBackend>::new(device, 3).load_record(record);
            Ok(model)
        }
        Err(e) => Err(format!("Failed to load model: {:?}", e).into()),
    }
}
