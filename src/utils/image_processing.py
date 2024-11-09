from PIL import Image
import numpy as np

def preprocess_image(image):
    """Enhanced preprocessing for various image formats"""
    try:
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
            
        if image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')
            
        # Apply image enhancement
        img_np = np.array(image)
        
        # Apply denoising if image is noisy
        if is_noisy(img_np):
            img_np = cv2.fastNlMeansDenoisingColored(img_np)
            
        # Apply contrast enhancement
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        img_np = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(img_np)
    except Exception as e:
        st.error(f"Error in image preprocessing: {str(e)}")
        return image