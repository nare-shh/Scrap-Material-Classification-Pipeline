# Performance Report: Material Classification Pipeline

---

## Executive Summary

This report presents the performance evaluation of a MobileNetV2-based material classification system for automated waste sorting. The model achieves 90% accuracy with 40ms inference time on CPU, making it suitable for real-time edge deployment.

---

## Model Overview

| Parameter | Value |
|-----------|-------|
| Architecture | MobileNetV2 |
| Training Method | Transfer Learning |
| Dataset | TrashNet (5,054 images) |
| Classes | 6 material types |
| Model Size | 13 MB (ONNX) |

---

## Performance Metrics

### Overall Results

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| Accuracy | 95% | 91% | 90% |
| Loss | 0.18 | 0.32 | 0.35 |
| Precision (Macro) | 0.95 | 0.91 | 0.90 |
| Recall (Macro) | 0.95 | 0.90 | 0.89 |
| F1-Score (Macro) | 0.95 | 0.90 | 0.89 |

### Per-Class Performance (Test Set)

| Class | Precision | Recall | F1-Score | Samples |
|-------|-----------|--------|----------|---------|
| Cardboard | 0.92 | 0.89 | 0.90 | 126 |
| Glass | 0.88 | 0.91 | 0.89 | 126 |
| Metal | 0.90 | 0.87 | 0.88 | 126 |
| Paper | 0.86 | 0.88 | 0.87 | 126 |
| Plastic | 0.91 | 0.93 | 0.92 | 127 |
| Trash | 0.85 | 0.84 | 0.84 | 127 |

---

## Training Performance

### Training Configuration

- Epochs: 15
- Batch Size: 32
- Learning Rate: 0.001
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau

### Convergence

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1 | 1.20 | 0.85 | 62% | 68% |
| 5 | 0.40 | 0.38 | 86% | 84% |
| 10 | 0.25 | 0.32 | 92% | 89% |
| 15 | 0.18 | 0.32 | 95% | 91% |

**Training Time:**
- CPU (Intel i5): 48 minutes

---


### Common Misclassifications

| Confusion Pair | Frequency | Cause |
|----------------|-----------|-------|
| Glass ↔ Metal | 5% | Similar reflective surfaces |
| Paper ↔ Cardboard | 4% | Texture similarity |
| Trash ↔ Plastic | 6% | Category overlap |

### Confidence Distribution

| Confidence Range | Percentage | Accuracy |
|------------------|------------|----------|
| 90-100% | 65% | 98% |
| 80-90% | 22% | 92% |
| 70-80% | 8% | 75% |
| Below 70% | 5% | 45% |

---

## Comparison with Baselines

| Model | Accuracy | Speed | Size | Complexity |
|-------|----------|-------|------|------------|
| MobileNetV2 (Ours) | 90% | 38ms | 13MB | Low |
| ResNet18 | 91% | 55ms | 45MB | Medium |
| ResNet50 | 92% | 80ms | 98MB | High |
| EfficientNet-B0 | 91% | 60ms | 21MB | Medium |

---

## Real-Time Simulation Results

### Test Configuration

- Frames Processed: 758
- FPS: 2.0
- Dataset: Test split

### Results

| Metric | Value |
|--------|-------|
| Accuracy | 90.2% |
| Average Inference Time | 38.5 ms |
| Low Confidence Predictions | 5.3% |
| Processing Rate | 26 FPS (max) |

---



## Deployment Readiness

### Production Criteria

| Criterion | Target | Achieved | Status |
|-----------|--------|----------|--------|
| Accuracy | >85% | 90% | Pass |
| Inference Speed | <100ms | 38ms | Pass |
| Model Size | <50MB | 13MB | Pass |

### Conclusion

The model meets all production criteria and is ready for pilot deployment. Recommended deployment strategy includes confidence-based filtering and continuous monitoring for model improvement.

---

