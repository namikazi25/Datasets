from transformers import BlipProcessor, BlipForConditionalGeneration

class InstructBLIP:
    """
    Wrapper for InstructBLIP model with consistency checking capabilities.
    """
    def __init__(self, model_name="Salesforce/instructblip-flan-t5-xl", device="cpu"):
        """
        Initialize InstructBLIP model and processor.

        Args:
            model_name (str): Pretrained model name from Hugging Face.
            device (str): Device to use ("cuda" or "cpu").
        """
        self.device = device
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name, ignore_mismatched_sizes=True).to(self.device)

    def check_consistency(self, image, text):
        """
        Check consistency between image and text.

        Args:
            image (PIL.Image): Input image.
            text (str): Input text.

        Returns:
            str: Generated consistency result.
        """
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        return self.processor.decode(outputs[0], skip_special_tokens=True)
