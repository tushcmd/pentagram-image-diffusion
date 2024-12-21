import modal
import io
from fastapi import Response, HTTPException, Query, Request
from datetime import datetime, timezone
import requests
import os


def download_model():
    from diffusers import AutoPipelineForText2Image
    import torch
    
    AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", 
        torch_dtype=torch.float16,
        variant="fp16"
    )
    
image = (modal.Image.debian_slim()
    .pip_install(
        "torch", 
        "fastapi[standard]", 
        "transformers", 
        "accelerate",  # Note: it's "accelerate" not "accelerator"
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
        from diffusers import AutoPipelineForText2Image
        import torch
        
        # self.pipe = AutoPipelineForText2Image(
        #     "stabilityai/sdxl-turbo", 
        #     torch_dtype=torch.float16,
        #     variant="fp16"
        # )
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", 
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        self.pipe.to("cuda")
        self.API_KEY = os.environ["API_KEY"]
        
        
    @modal.web_endpoint()
    def generate(self, request: Request, prompt: str = Query(..., description="The prompt for image generation")):
        
        
        api_key = request.headers.get("x-api-key")
        
        if api_key != self.API_KEY:
            raise HTTPException(
                status_code=401, 
                detail="Unauthorized"
            )
        
        image = self.pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        
        return Response(content=buffer.getvalue(), media_type="image/jpeg")
    
    @modal.web_endpoint()
    def health(self):
        """Lightweight health check endpoint for keeping the container warm"""
        return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

# warm-keeping function that runs every 5 minutes
@app.function(
    schedule=modal.Cron("*/5 * * * *"),
    secrets=[modal.Secret.from_name("API_KEY")]
)

def keep_warm():
    # pass
    health_url = "https://tushcmd--stable-diffusion-modal-health.modal.run"
    generate_url = "https://tushcmd--stable-diffusion-modal-generate.modal.run"
    
    #First check health endpoint (no API key required)
    health_response = requests.get(health_url)
    print(f"Health check at: {health_response.json()['timestamp']}")
    
    # Then make a test request to the generate endpoint
    headers = {"x-api-key": os.environ["API_KEY"]}
    generate_response = requests.get(generate_url, headers=headers)
    print(f"Generate endpoint tested successfully: {datetime.now().isoformat()}")
   