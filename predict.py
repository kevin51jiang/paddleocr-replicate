# Prediction interface for Cog ⚙️
# https://cog.run/python

from cog import BasePredictor, Input, Path
from paddleocr import PaddleOCR
import json
import tempfile




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
        image: Path = Input(description="Input image"),
    ) -> Path:
        """Run a single prediction on the model"""
        
        result = self.ocr.predict(str(image))

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
