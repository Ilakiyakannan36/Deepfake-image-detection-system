import gradio as gr
from src.inference import DeepfakeDetectorInference
from src.utils import load_config

def create_web_app():
    config = load_config('config.yaml')
    detector = DeepfakeDetectorInference(config['inference']['model_path'])
    
    def predict_image(image):
        # Convert Gradio image to OpenCV format
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            result = detector.predict(tmp.name)
            os.unlink(tmp.name)
            
        if 'error' in result:
            return {"Error": result['error']}
        
        return {
            "Prediction": result['prediction'],
            "Confidence": f"{result['confidence']:.3f}",
            "Real Probability": f"{result['real_probability']:.3f}",
            "Fake Probability": f"{result['fake_probability']:.3f}"
        }
    
    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=gr.Label(label="Detection Results"),
        title="DeepGuard - Deepfake Detection System",
        description="Upload an image to detect AI-generated or manipulated content",
        examples=[["example_real.jpg"], ["example_fake.jpg"]]
    )
    
    return interface

if __name__ == "__main__":
    app = create_web_app()
    app.launch(share=True)