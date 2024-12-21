# Code step by step

Complete code for deployement

```py
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


```

With error handling

```py
import modal
import io
from fastapi import Response, HTTPException, Query, Request
from datetime import datetime, timezone
import requests
import os

image = (modal.Image.debian_slim()
    .pip_install(
        "torch",
        "fastapi[standard]",
        "transformers",
        "accelerate",
        "diffusers",
        "requests"
    ))

app = modal.App("stable-diffusion", image=image)

@app.cls(
    image=image,
    gpu="A10G",
    secrets=[modal.Secret.from_name("API_KEY")]
)
class Modal:

    @modal.build()
    @modal.enter()
    def load_weights(self):
        from diffusers import AutoPipelineForText2Image
        import torch

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

        try:
            image = self.pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")

            return Response(content=buffer.getvalue(), media_type="image/jpeg")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating image: {str(e)}"
            )
```

Added cron jobs to keep spp warm
