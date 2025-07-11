# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
from paddleocr import PaddleOCR
import json
import tempfile
import base64
import os




class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False, 
            use_doc_unwarping=False, 
            use_textline_orientation=False) 

    def predict(
        self,
        image: str = Input(description="Base64 encoded image"),
    ) -> Path:
        """Run a single prediction on the model"""
        
        # Handle data URI prefix if present (e.g., data:image/png;base64,)
        if image.startswith('data:'):
            # Find the comma that separates the header from the base64 data
            comma_index = image.find(',')
            if comma_index != -1:
                image = image[comma_index + 1:]
        
        # Decode base64 string to binary data
        image_data = base64.b64decode(image)
        
        # Create a temporary file to save the decoded image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img_file:
            temp_img_file.write(image_data)
            temp_img_path = temp_img_file.name
        
        try:
            result = self.ocr.predict(temp_img_path)
        finally:
            # Clean up the temporary image file
            os.unlink(temp_img_path)

        # Extract the first result (PaddleOCR returns a list)
        if result and len(result) > 0:
            res = result[0]
            
            # Create a temporary file to save the JSON result
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                # Save the result to JSON
                res.save_to_json(f.name)
                temp_json_path = f.name
            
            return Path(temp_json_path)
        else:
            # If no results, return empty JSON
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump({"error": "No OCR results found"}, f)
                temp_json_path = f.name
            
            return Path(temp_json_path)
