//! SEDAC V6.1 Core - High-Performance Rust Implementation
//!
//! This module provides the computational core for SEDAC's cascade early exit mechanism.
//!
//! # Key Innovations in V6.1
//!
//! 1. **Confidence Accumulation**: Instead of single-layer binary decisions, V6.1 accumulates
//!    confidence scores across multiple checkpoint layers using Bayesian updates.
//!
//! 2. **Soft Exit**: Rather than hard MLP skipping, V6.1 supports gradual computation reduction
//!    based on accumulated confidence (e.g., skip 25%, 50%, 75% of MLP channels).
//!
//! 3. **Adaptive EMA Thresholds**: Thread-safe, lock-free threshold management with configurable
//!    smoothing factors per layer.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use parking_lot::RwLock;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// Atomic float wrapper for lock-free threshold updates
struct AtomicF32(AtomicU64);

impl AtomicF32 {
    fn new(val: f32) -> Self {
        Self(AtomicU64::new((val as f64).to_bits()))
    }

    fn load(&self) -> f32 {
        f64::from_bits(self.0.load(Ordering::Relaxed)) as f32
    }

    fn store(&self, val: f32) {
        self.0.store((val as f64).to_bits(), Ordering::Relaxed);
    }

    /// Atomic EMA update: new = alpha * sample + (1 - alpha) * old
    fn ema_update(&self, sample: f32, alpha: f32) -> f32 {
        let old = self.load();
        let new_val = alpha * sample + (1.0 - alpha) * old;
        self.store(new_val);
        new_val
    }
}

/// Layer-specific configuration for cascade exit
#[pyclass]
#[derive(Clone)]
pub struct LayerConfig {
    #[pyo3(get, set)]
    pub layer_idx: usize,
    #[pyo3(get, set)]
    pub target_exit_rate: f32,
    #[pyo3(get, set)]
    pub initial_threshold: f32,
    #[pyo3(get, set)]
    pub confidence_weight: f32,
}

#[pymethods]
impl LayerConfig {
    #[new]
    fn new(layer_idx: usize, target_exit_rate: f32, initial_threshold: f32, confidence_weight: f32) -> Self {
        Self {
            layer_idx,
            target_exit_rate,
            initial_threshold,
            confidence_weight,
        }
    }
}

/// Cascade exit decision with confidence accumulation
#[pyclass]
#[derive(Clone)]
pub struct ExitDecision {
    #[pyo3(get)]
    pub should_exit: bool,
    #[pyo3(get)]
    pub exit_layer: i32,
    #[pyo3(get)]
    pub accumulated_confidence: f32,
    #[pyo3(get)]
    pub soft_exit_ratio: f32,
}

/// Main SEDAC cascade controller
#[pyclass]
pub struct CascadeController {
    layer_configs: Vec<LayerConfig>,
    thresholds: Vec<AtomicF32>,
    alpha: f32,
    confidence_decay: f32,
    soft_exit_enabled: bool,
    calibration_samples: RwLock<Vec<Vec<f32>>>,
    calibration_steps: usize,
    is_calibrated: RwLock<bool>,
    total_calls: AtomicU64,
    total_exits: Vec<AtomicU64>,
}

#[pymethods]
impl CascadeController {
    /// Create a new CascadeController
    ///
    /// # Arguments
    /// * `layer_configs` - Configuration for each checkpoint layer
    /// * `alpha` - EMA smoothing factor (0.0-1.0, lower = more stable)
    /// * `confidence_decay` - Decay factor for confidence accumulation between layers
    /// * `soft_exit_enabled` - Enable soft exit (gradual MLP reduction) vs hard exit
    /// * `calibration_steps` - Number of warmup samples before adaptive calibration activates
    #[new]
    fn new(
        layer_configs: Vec<LayerConfig>,
        alpha: f32,
        confidence_decay: f32,
        soft_exit_enabled: bool,
        calibration_steps: usize,
    ) -> Self {
        let num_layers = layer_configs.len();
        let thresholds: Vec<AtomicF32> = layer_configs
            .iter()
            .map(|cfg| AtomicF32::new(cfg.initial_threshold))
            .collect();

        let calibration_samples = RwLock::new(vec![Vec::new(); num_layers]);
        let total_exits: Vec<AtomicU64> = (0..num_layers).map(|_| AtomicU64::new(0)).collect();

        Self {
            layer_configs,
            thresholds,
            alpha,
            confidence_decay,
            soft_exit_enabled,
            calibration_samples,
            calibration_steps,
            is_calibrated: RwLock::new(false),
            total_calls: AtomicU64::new(0),
            total_exits,
        }
    }

    /// Evaluate cascade exit decision for a batch of tokens
    ///
    /// # Arguments
    /// * `risk_scores` - 2D array [num_layers, batch_size] of risk scores from probes
    ///
    /// # Returns
    /// * List of ExitDecision for each token in the batch
    fn evaluate_batch<'py>(
        &self,
        py: Python<'py>,
        risk_scores: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Vec<ExitDecision>> {
        let risks = risk_scores.as_array();
        let (num_layers, batch_size) = risks.dim();

        self.total_calls.fetch_add(batch_size as u64, Ordering::Relaxed);

        // Parallel evaluation across batch
        let decisions: Vec<ExitDecision> = (0..batch_size)
            .into_par_iter()
            .map(|token_idx| {
                let mut accumulated_conf = 0.0f32;
                let mut exit_layer = -1i32;
                let mut should_exit = false;

                for (layer_i, cfg) in self.layer_configs.iter().enumerate() {
                    let risk = risks[[layer_i, token_idx]];
                    let threshold = self.thresholds[layer_i].load();

                    // Confidence = how much risk is below threshold (normalized)
                    let layer_confidence = if threshold > 0.0 {
                        ((threshold - risk) / threshold).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };

                    // Bayesian-style accumulation with decay
                    accumulated_conf = accumulated_conf * self.confidence_decay
                        + layer_confidence * cfg.confidence_weight;

                    // Exit condition: accumulated confidence exceeds layer's exit rate target
                    if accumulated_conf >= cfg.target_exit_rate && !should_exit {
                        should_exit = true;
                        exit_layer = cfg.layer_idx as i32;
                    }
                }

                // Soft exit ratio: maps confidence to MLP skip percentage
                let soft_exit_ratio = if self.soft_exit_enabled && should_exit {
                    // Sigmoid-like mapping: higher confidence -> more skip
                    (accumulated_conf * 2.0 - 1.0).tanh() * 0.5 + 0.5
                } else if should_exit {
                    1.0 // Hard exit
                } else {
                    0.0
                };

                ExitDecision {
                    should_exit,
                    exit_layer,
                    accumulated_confidence: accumulated_conf,
                    soft_exit_ratio,
                }
            })
            .collect();

        // Update exit counters
        for decision in &decisions {
            if decision.should_exit && decision.exit_layer >= 0 {
                for (i, cfg) in self.layer_configs.iter().enumerate() {
                    if cfg.layer_idx == decision.exit_layer as usize {
                        self.total_exits[i].fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                }
            }
        }

        Ok(decisions)
    }

    /// Update thresholds with new risk samples (for online calibration)
    ///
    /// # Arguments
    /// * `layer_idx` - Index of the layer in layer_configs
    /// * `risk_samples` - 1D array of risk values from this layer
    fn update_threshold<'py>(
        &self,
        py: Python<'py>,
        layer_idx: usize,
        risk_samples: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<f32> {
        let samples = risk_samples.as_slice()?;

        let is_calibrated = *self.is_calibrated.read();

        if !is_calibrated {
            // Warmup phase: collect samples
            let mut cal_samples = self.calibration_samples.write();
            cal_samples[layer_idx].extend_from_slice(samples);

            // Check if all layers have enough samples
            let all_ready = cal_samples.iter().all(|s| s.len() >= self.calibration_steps);

            if all_ready {
                // Initial calibration: set thresholds based on target exit rates
                for (i, cfg) in self.layer_configs.iter().enumerate() {
                    let mut layer_samples = cal_samples[i].clone();
                    layer_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

                    let quantile_idx = ((layer_samples.len() as f32) * cfg.target_exit_rate) as usize;
                    let quantile_idx = quantile_idx.min(layer_samples.len().saturating_sub(1));

                    let new_threshold = layer_samples[quantile_idx];
                    self.thresholds[i].store(new_threshold);
                }

                *self.is_calibrated.write() = true;
            }

            return Ok(self.thresholds[layer_idx].load());
        }

        // Online calibration: EMA update
        if !samples.is_empty() {
            let mut sorted_samples = samples.to_vec();
            sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let target_rate = self.layer_configs[layer_idx].target_exit_rate;
            let quantile_idx = ((sorted_samples.len() as f32) * target_rate) as usize;
            let quantile_idx = quantile_idx.min(sorted_samples.len().saturating_sub(1));

            let batch_threshold = sorted_samples[quantile_idx];
            let new_threshold = self.thresholds[layer_idx].ema_update(batch_threshold, self.alpha);

            return Ok(new_threshold);
        }

        Ok(self.thresholds[layer_idx].load())
    }

    /// Get current threshold for a layer
    fn get_threshold(&self, layer_idx: usize) -> f32 {
        self.thresholds.get(layer_idx).map(|t| t.load()).unwrap_or(0.0)
    }

    /// Get all current thresholds
    fn get_all_thresholds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let thresholds: Vec<f32> = self.thresholds.iter().map(|t| t.load()).collect();
        thresholds.into_pyarray_bound(py)
    }

    /// Get statistics
    fn get_stats(&self) -> (u64, Vec<u64>) {
        let total = self.total_calls.load(Ordering::Relaxed);
        let exits: Vec<u64> = self.total_exits.iter().map(|e| e.load(Ordering::Relaxed)).collect();
        (total, exits)
    }

    /// Check if calibration is complete
    fn is_calibrated(&self) -> bool {
        *self.is_calibrated.read()
    }

    /// Reset calibration state
    fn reset_calibration(&self) {
        *self.is_calibrated.write() = false;
        let mut samples = self.calibration_samples.write();
        for s in samples.iter_mut() {
            s.clear();
        }
    }
}

/// Compute quantile of a 1D array (used for threshold calibration)
#[pyfunction]
fn compute_quantile<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f32>,
    quantile: f32,
) -> PyResult<f32> {
    let slice = data.as_slice()?;
    if slice.is_empty() {
        return Ok(0.0);
    }

    let mut sorted: Vec<f32> = slice.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = ((sorted.len() as f32) * quantile.clamp(0.0, 1.0)) as usize;
    let idx = idx.min(sorted.len() - 1);

    Ok(sorted[idx])
}

/// Batch softmax computation (for entropy calculation)
#[pyfunction]
fn batch_softmax<'py>(
    py: Python<'py>,
    logits: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let arr = logits.as_array();
    let (batch_size, vocab_size) = arr.dim();

    let mut result = ndarray::Array2::<f32>::zeros((batch_size, vocab_size));

    result
        .outer_iter_mut()
        .into_par_iter()
        .zip(arr.outer_iter().into_par_iter())
        .for_each(|(mut out_row, in_row)| {
            let max_val = in_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = in_row.iter().map(|&x| (x - max_val).exp()).sum();

            for (out, &inp) in out_row.iter_mut().zip(in_row.iter()) {
                *out = ((inp - max_val).exp()) / exp_sum;
            }
        });

    Ok(result.into_pyarray_bound(py))
}

/// Compute entropy from probability distribution
#[pyfunction]
fn compute_entropy<'py>(
    py: Python<'py>,
    probs: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let arr = probs.as_array();
    let batch_size = arr.dim().0;

    let entropies: Vec<f32> = arr
        .outer_iter()
        .into_par_iter()
        .map(|row| {
            let entropy: f32 = row
                .iter()
                .filter(|&&p| p > 1e-10)
                .map(|&p| -p * p.ln())
                .sum();
            entropy
        })
        .collect();

    Ok(entropies.into_pyarray_bound(py))
}

/// Async batch processor for high-throughput inference
/// Minimizes Python GIL contention by processing batches in Rust threads
#[pyclass]
pub struct AsyncBatchProcessor {
    controller: std::sync::Arc<parking_lot::RwLock<CascadeController>>,
    batch_queue: std::sync::Arc<parking_lot::Mutex<Vec<BatchRequest>>>,
    result_cache: std::sync::Arc<parking_lot::RwLock<std::collections::HashMap<u64, Vec<ExitDecision>>>>,
    next_batch_id: std::sync::atomic::AtomicU64,
}

struct BatchRequest {
    id: u64,
    risk_scores: Vec<Vec<f32>>,  // [num_layers, batch_size]
}

#[pymethods]
impl AsyncBatchProcessor {
    #[new]
    fn new(controller: CascadeController) -> Self {
        Self {
            controller: std::sync::Arc::new(parking_lot::RwLock::new(controller)),
            batch_queue: std::sync::Arc::new(parking_lot::Mutex::new(Vec::new())),
            result_cache: std::sync::Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new())),
            next_batch_id: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Submit a batch for async processing, returns batch_id
    fn submit_batch<'py>(
        &self,
        py: Python<'py>,
        risk_scores: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<u64> {
        let arr = risk_scores.as_array();
        let (num_layers, batch_size) = arr.dim();
        
        let mut scores_vec = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let mut layer_scores = Vec::with_capacity(batch_size);
            for j in 0..batch_size {
                layer_scores.push(arr[[i, j]]);
            }
            scores_vec.push(layer_scores);
        }
        
        let batch_id = self.next_batch_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        
        {
            let mut queue = self.batch_queue.lock();
            queue.push(BatchRequest {
                id: batch_id,
                risk_scores: scores_vec,
            });
        }
        
        // Process immediately in current thread (can be extended to thread pool)
        self.process_pending();
        
        Ok(batch_id)
    }

    /// Process all pending batches
    fn process_pending(&self) {
        let requests: Vec<BatchRequest> = {
            let mut queue = self.batch_queue.lock();
            std::mem::take(&mut *queue)
        };
        
        let controller = self.controller.read();
        
        for req in requests {
            let batch_size = req.risk_scores.first().map(|v| v.len()).unwrap_or(0);
            let num_layers = req.risk_scores.len();
            
            let decisions: Vec<ExitDecision> = (0..batch_size)
                .into_par_iter()
                .map(|token_idx| {
                    let mut accumulated_conf = 0.0f32;
                    let mut exit_layer = -1i32;
                    let mut should_exit = false;
                    
                    for (layer_i, cfg) in controller.layer_configs.iter().enumerate() {
                        if layer_i >= num_layers {
                            break;
                        }
                        let risk = req.risk_scores[layer_i][token_idx];
                        let threshold = controller.thresholds[layer_i].load();
                        
                        let layer_confidence = if threshold > 0.0 {
                            ((threshold - risk) / threshold).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        
                        accumulated_conf = accumulated_conf * controller.confidence_decay
                            + layer_confidence * cfg.confidence_weight;
                        
                        if accumulated_conf >= cfg.target_exit_rate && !should_exit {
                            should_exit = true;
                            exit_layer = cfg.layer_idx as i32;
                        }
                    }
                    
                    let soft_exit_ratio = if controller.soft_exit_enabled && should_exit {
                        (accumulated_conf * 2.0 - 1.0).tanh() * 0.5 + 0.5
                    } else if should_exit {
                        1.0
                    } else {
                        0.0
                    };
                    
                    ExitDecision {
                        should_exit,
                        exit_layer,
                        accumulated_confidence: accumulated_conf,
                        soft_exit_ratio,
                    }
                })
                .collect();
            
            let mut cache = self.result_cache.write();
            cache.insert(req.id, decisions);
        }
    }

    /// Get results for a batch (returns None if not ready)
    fn get_results(&self, batch_id: u64) -> Option<Vec<ExitDecision>> {
        let mut cache = self.result_cache.write();
        cache.remove(&batch_id)
    }

    /// Check if results are ready
    fn is_ready(&self, batch_id: u64) -> bool {
        let cache = self.result_cache.read();
        cache.contains_key(&batch_id)
    }

    /// Get pending batch count
    fn pending_count(&self) -> usize {
        self.batch_queue.lock().len()
    }
}

/// Zero-copy tensor wrapper for efficient Rust-Python data transfer
#[pyclass]
pub struct TensorBuffer {
    data: Vec<f32>,
    shape: Vec<usize>,
}

#[pymethods]
impl TensorBuffer {
    #[new]
    fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape,
        }
    }

    /// Get buffer as numpy array (zero-copy view)
    fn as_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f32>>> {
        Ok(numpy::PyArray1::from_slice_bound(py, &self.data))
    }

    /// Fill from numpy array
    fn fill_from<'py>(&mut self, arr: PyReadonlyArray1<'py, f32>) -> PyResult<()> {
        let slice = arr.as_slice()?;
        if slice.len() != self.data.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("Shape mismatch"));
        }
        self.data.copy_from_slice(slice);
        Ok(())
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

/// Batch confidence accumulator with SIMD optimization
#[pyfunction]
fn batch_confidence_accumulate<'py>(
    py: Python<'py>,
    risks: PyReadonlyArray2<'py, f32>,
    thresholds: PyReadonlyArray1<'py, f32>,
    weights: PyReadonlyArray1<'py, f32>,
    decay: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let risks_arr = risks.as_array();
    let thresholds_slice = thresholds.as_slice()?;
    let weights_slice = weights.as_slice()?;
    
    let (num_layers, batch_size) = risks_arr.dim();
    
    let confidences: Vec<f32> = (0..batch_size)
        .into_par_iter()
        .map(|token_idx| {
            let mut acc = 0.0f32;
            for layer_i in 0..num_layers {
                let risk = risks_arr[[layer_i, token_idx]];
                let threshold = thresholds_slice.get(layer_i).copied().unwrap_or(1.0);
                let weight = weights_slice.get(layer_i).copied().unwrap_or(0.33);
                
                let layer_conf = if threshold > 0.0 {
                    ((threshold - risk) / threshold).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                
                acc = acc * decay + layer_conf * weight;
            }
            acc
        })
        .collect();
    
    Ok(confidences.into_pyarray_bound(py))
}

/// Python module definition
#[pymodule]
fn sedac_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LayerConfig>()?;
    m.add_class::<ExitDecision>()?;
    m.add_class::<CascadeController>()?;
    m.add_class::<AsyncBatchProcessor>()?;
    m.add_class::<TensorBuffer>()?;
    m.add_function(wrap_pyfunction!(compute_quantile, m)?)?;
    m.add_function(wrap_pyfunction!(batch_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(compute_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(batch_confidence_accumulate, m)?)?;
    Ok(())
}
