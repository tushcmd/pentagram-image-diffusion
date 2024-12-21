import modal
import io
from fastapi import Response, HTTPException, Query, Request
from datetime import datetime, timezone
import requests
import os

def download_model():
    from diffusers import AutoPipelineForText2Image
    import torch
    
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", 
        torch_dtype=torch.float16,
        variant="fp16"
    )
    return pipe  # Return the pipeline to ensure it's actually downloaded

image = (modal.Image.debian_slim()
    .pip_install(
        "torch",
        "fastapi[standard]", 
        "transformers", 
        "accelerate",
        "diffusers", 
        "requests"
    )
    .run_function(download_model))

app = modal.App("stable-diffusion", image=image)

@app.cls(
    image=image,
    gpu="A10G",
    container_idle_timeout=300,
    secrets=[modal.Secret.from_name("API_KEY")]
)
class Modal:
    
    @modal.build()
    @modal.enter()
    def load_weights(self):
        try:
            from diffusers import AutoPipelineForText2Image
            import torch
            
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo", 
                torch_dtype=torch.float16,
                variant="fp16"
            )
            
            self.pipe.to("cuda")
            self.API_KEY = os.environ["API_KEY"]
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    @modal.web_endpoint()
    def generate(self, request: Request, prompt: str = Query(..., description="The prompt for image generation")):
        try:
            api_key = request.headers.get("x-api-key")
            
            if not api_key:
                raise HTTPException(
                    status_code=401,
                    detail="API key missing"
                )
            
            if api_key != self.API_KEY:
                raise HTTPException(
                    status_code=401, 
                    detail="Unauthorized"
                )
            
            if not prompt or len(prompt.strip()) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Prompt cannot be empty"
                )
            
            image = self.pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
            
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            
            return Response(content=buffer.getvalue(), media_type="image/jpeg")
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating image: {str(e)}"
            )
    
    @modal.web_endpoint()
    def health(self):
        """Lightweight health check endpoint for keeping the container warm"""
        try:
            return {
                "status": "healthy", 
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Health check failed: {str(e)}"
            )

@app.function(
    schedule=modal.Cron("*/5 * * * *"),
    secrets=[modal.Secret.from_name("API_KEY")]
)
def keep_warm():
    try:
        health_url = "https://tushcmd--stable-diffusion-modal-health.modal.run"
        generate_url = "https://tushcmd--stable-diffusion-modal-generate.modal.run"
        
        # Health check
        health_response = requests.get(health_url)
        health_response.raise_for_status()  # Raise exception for non-200 status codes
        print(f"Health check at: {health_response.json()['timestamp']}")
        
        # Test generate endpoint with a simple prompt
        headers = {"x-api-key": os.environ["API_KEY"]}
        params = {"prompt": "test image"}  # Add prompt parameter
        generate_response = requests.get(generate_url, headers=headers, params=params)
        generate_response.raise_for_status()
        print(f"Generate endpoint tested successfully: {datetime.now(timezone.utc).isoformat()}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error in keep_warm function: {str(e)}")
        raise