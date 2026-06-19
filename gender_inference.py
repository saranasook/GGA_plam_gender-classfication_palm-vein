import torch
from torchvision import transforms
from PIL import Image
import os

# Import your custom architecture from your model file
from gender_model import DenseNet161_GGA_Binary

def predict_single_image(image_path, model_path, device=None):
    """
    Loads the trained GGA binary model and runs inference on a single raw image.
    """
    # 1. Determine device setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Define inference transforms matching your exact training configuration 
    # (Excluding random augmentations like horizontal flip)
    inference_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 3. Check and open image file
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Provided image path not found: {image_path}")
        
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms and add batch dimension: shape becomes [1, 3, 224, 224]
    input_tensor = inference_transforms(image).unsqueeze(0).to(device)
    
    # 4. Initialize model instance and map state dict
    print("Initializing DenseNet161_GGA_Binary network...")
    model = DenseNet161_GGA_Binary()
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint weight path not found: {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 5. Run forward prediction pass
    with torch.no_grad():
        output = model(input_tensor)
        
        # Pull raw probability scalar out of tensor
        probability = output.item() 
        
        # Consistent with your code logic map: 0.0 = Male ('M'), 1.0 = Female ('F')
        pred_label = 1 if probability > 0.5 else 0
        gender_string = "Female (F)" if pred_label == 1 else "Male (M)"
        
    # Print clean diagnostic terminal readout
    print("-" * 40)
    print(f"Inference Target: {os.path.basename(image_path)}")
    print(f"Raw Sigmoid Output Score: {probability:.4f}")
    print(f"Predicted Class Result  : {gender_string}")
    print("-" * 40)
    
    return {
        'probability': probability,
        'label_id': pred_label,
        'gender': gender_string
    }

if __name__ == "__main__":
    # --- Quick Execution Config ---
    # Update these string parameters with your local diagnostic targets
    MODEL_WEIGHTS_PATH = "/path/best.pt"
    TEST_IMAGE_PATH = "/project/datasets/VERA-Palmvein/sample.png"
    
    # Run targeted execution block
    try:
        result = predict_single_image(
            image_path=TEST_IMAGE_PATH,
            model_path=MODEL_WEIGHTS_PATH
        )
    except Exception as e:
        print(f"Inference Pipeline Error: {e}")
