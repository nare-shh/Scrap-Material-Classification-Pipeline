import onnxruntime as ort
import numpy as np
from PIL import Image
import time
import os


class ONNXInferenceEngine:
    
    def __init__(self, model_path='models/material_classifier.onnx', 
                 class_names_path='models/class_names.txt'):
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}\nPlease run training and export first!")
        
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        if not os.path.exists(class_names_path):
            raise FileNotFoundError(f"Class names not found: {class_names_path}")
            
        with open(class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        print(f"Model loaded: {len(self.class_names)} classes")
        print(f"Classes: {', '.join(self.class_names)}")
    
    def preprocess_image(self, image_path_or_pil):
    
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil)
        else:
            image = image_path_or_pil
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        
        image_array = np.array(image).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array.astype(np.float32)
    
    def predict(self, image_path_or_pil, return_all_probs=False):
        
        start_time = time.time()
        input_tensor = self.preprocess_image(image_path_or_pil)
        preprocess_time = time.time() - start_time
        
        start_time = time.time()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        inference_time = time.time() - start_time
        
        logits = outputs[0][0]
        
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / exp_logits.sum()
        
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'predicted_idx': int(predicted_idx),
            'inference_time_ms': inference_time * 1000,
            'preprocess_time_ms': preprocess_time * 1000,
            'total_time_ms': (preprocess_time + inference_time) * 1000
        }
        
        if return_all_probs:
            result['all_probabilities'] = {
                self.class_names[i]: float(probabilities[i]) 
                for i in range(len(self.class_names))
            }
        
        return result
    
    def predict_batch(self, image_paths):
        
        results = []
        for image_path in image_paths:
            result = self.predict(image_path)
            results.append(result)
        return results


def test_inference():
    from datasets import load_dataset
    
    engine = ONNXInferenceEngine()
    
    ds = load_dataset("garythung/trashnet")
    test_dataset = ds['test']
    
    print("\nRunning predictions on 5 test samples:\n")
    
    for i in range(5):
        sample = test_dataset[i]
        image = sample['image']
        true_label = sample['label']
        true_class = test_dataset.features['label'].names[true_label]
        
        result = engine.predict(image, return_all_probs=True)
        
        print(f"Sample {i+1}:")
        print(f"  True Label: {true_class}")
        print(f"  Predicted: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Inference Time: {result['inference_time_ms']:.2f} ms")
        print(f"  Total Time: {result['total_time_ms']:.2f} ms")
        
        match = "Success" if result['predicted_class'] == true_class else "fail"
        print(f"  Status: {match}")
        print()


if __name__ == "__main__":
    test_inference()