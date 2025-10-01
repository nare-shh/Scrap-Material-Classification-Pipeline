import time
import csv
import os
import sys
from datetime import datetime
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.inference import ONNXInferenceEngine
except ImportError:
    from inference import ONNXInferenceEngine

import argparse


class ConveyorSimulator:
    
    def __init__(self, model_path='models/material_classifier.onnx',
                 class_names_path='models/class_names.txt',
                 confidence_threshold=0.70,
                 output_csv='results/classification_results.csv'):
        
        self.engine = ONNXInferenceEngine(model_path, class_names_path)
        self.confidence_threshold = confidence_threshold
        self.output_csv = output_csv
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        self._initialize_csv()
        
        print(f"Conveyor simulator initialized")
        print(f"Low confidence threshold: {confidence_threshold:.0%}")
        print(f"Results will be saved to: {output_csv}\n")
    
    def _initialize_csv(self):
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame_id',
                'predicted_class',
                'true_class',
                'correct',
                'inference_time_ms'
            ])
    
    def _log_to_csv(self, frame_id, result, true_class):
        is_correct = result['predicted_class'] == true_class
        
        with open(self.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_id,
                result['predicted_class'],
                true_class,
                'YES' if is_correct else 'NO',
                f"{result['inference_time_ms']:.2f}"
            ])
    
    def _print_result(self, frame_id, result, true_class=None):
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        output = f"[{timestamp}] Frame #{frame_id:04d} | "
        output += f"Predicted: {result['predicted_class']:10s} | "
        output += f"Confidence: {result['confidence']:6.2%} | "
        output += f"Time: {result['inference_time_ms']:5.1f}ms"
        
        if true_class:
            match = "✅" if result['predicted_class'] == true_class else "❌"
            output += f" | True: {true_class:10s} {match}"
        
        if result['confidence'] < self.confidence_threshold:
            output += " ⚠️  LOW CONFIDENCE!"
        
        print(output)
    
    def run_simulation(self, fps=2, max_frames=None, use_test_set=True):
        
        print("=" * 100)
        print(f"FPS: {fps} | Interval: {1/fps:.2f}s per frame")
        print(f"Dataset: {'Test' if use_test_set else 'Train'} set")
        print("=" * 100)
        print()
        
        ds = load_dataset("garythung/trashnet")
        
        if use_test_set:
            train_val_test = ds['train'].train_test_split(test_size=0.3, seed=42)
            val_test = train_val_test['test'].train_test_split(test_size=0.5, seed=42)
            dataset = val_test['test']  # Use test split
        else:
            dataset = ds['train']
        
        class_names = ds['train'].features['label'].names
        
        total_frames = len(dataset) if max_frames is None else min(max_frames, len(dataset))
        
        print(f"Total frames to process: {total_frames}\n")
        
        stats = {
            'total': 0,
            'correct': 0,
            'low_confidence': 0,
            'total_inference_time': 0
        }
        
        interval = 1.0 / fps
        
        try:
            for frame_id in range(total_frames):
                frame_start = time.time()
                
                sample = dataset[frame_id]
                image = sample['image']
                true_label = sample['label']
                true_class = class_names[true_label]
                
                result = self.engine.predict(image)
                
                stats['total'] += 1
                stats['total_inference_time'] += result['inference_time_ms']
                
                if result['predicted_class'] == true_class:
                    stats['correct'] += 1
                
                if result['confidence'] < self.confidence_threshold:
                    stats['low_confidence'] += 1
                
                self._log_to_csv(frame_id, result, true_class)
                self._print_result(frame_id, result, true_class)
                
                elapsed = time.time() - frame_start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\n\n Simulation interrupted by user")
        
        self._print_summary(stats)
    
    def _print_summary(self, stats):
        print("\n" + "=" * 100)
        print("SIMULATION SUMMARY")
        print("=" * 100)
        
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        avg_time = (stats['total_inference_time'] / stats['total']) if stats['total'] > 0 else 0
        low_conf_pct = (stats['low_confidence'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        print(f" Total Frames Processed: {stats['total']}")
        print(f" Accuracy: {accuracy:.2f}% ({stats['correct']}/{stats['total']})")
        print(f"  Low Confidence Predictions: {stats['low_confidence']} ({low_conf_pct:.1f}%)")
        print(f" Average Inference Time: {avg_time:.2f} ms")
        print(f" Results saved to: {self.output_csv}")
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Conveyor Belt Material Classification Simulator')
    
    parser.add_argument('--fps', type=float, default=2.0,
                       help='Frames per second (default: 2.0)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process (default: all)')
    parser.add_argument('--threshold', type=float, default=0.70,
                       help='Low confidence threshold (default: 0.70)')
    parser.add_argument('--train', action='store_true',
                       help='Use training set instead of test set')
    parser.add_argument('--model', type=str, default='models/material_classifier.onnx',
                       help='Path to ONNX model')
    parser.add_argument('--output', type=str, default='results/classification_results.csv',
                       help='Output CSV path')
    
    args = parser.parse_args()
    
    simulator = ConveyorSimulator(
        model_path=args.model,
        confidence_threshold=args.threshold,
        output_csv=args.output
    )
    
    simulator.run_simulation(
        fps=args.fps,
        max_frames=args.max_frames,
        use_test_set=not args.train
    )


if __name__ == "__main__":
    main()