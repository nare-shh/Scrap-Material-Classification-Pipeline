import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import os
from model_training import MaterialClassifier


def export_to_onnx(model_path='models/best_model.pth', 
                   output_path='models/material_classifier.onnx'):
    
    
    checkpoint = torch.load(model_path, map_location='cpu')
    class_names = checkpoint['class_names']
    
    model = MaterialClassifier(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded (Val Acc: {checkpoint['val_acc']:.2f}%)")
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"Exporting to ONNX")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX model saved to {output_path}")
    
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")
    

    print("Testing ONNX inference")
    ort_session = ort.InferenceSession(output_path)
    
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    with torch.no_grad():
        torch_output = model(dummy_input)
    
    np.testing.assert_allclose(
        torch_output.numpy(), 
        ort_outputs[0], 
        rtol=1e-03, 
        atol=1e-05
    )
    print("ONNX inference matches PyTorch output")
    
    model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Model size: {model_size_mb:.2f} MB")
    
    class_names_path = 'models/class_names.txt'
    with open(class_names_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"Class names saved to {class_names_path}")
    
    return output_path


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    
    onnx_path = export_to_onnx()
    print(f"\nExport complete. ONNX model ready at: {onnx_path}")