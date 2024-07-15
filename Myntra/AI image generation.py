import tkinter as tk
import customtkinter as ctk
from PIL import ImageTk, Image
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Replace 'your_auth_token' with your actual Hugging Face auth token
auth_token = 'abcdef0123456789abcdef0123456789abcdef01'

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

# Create prompt entry field
prompt = ctk.CTkEntry(app, height=40, width=512, fg_color="white")
prompt.place(x=10, y=10)

# Create label to display image
lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

# Load model
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model without specifying torch_dtype if using CPU
if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float32, use_auth_token=auth_token)
else:
    pipe = StableDiffusionPipeline.from_pretrained(modelid, use_auth_token=auth_token)
pipe.to(device)

def generate():
    prompt_text = prompt.get()
    # Using autocast only if the device is CUDA
    if device == "cuda":
        with autocast(device, dtype=torch.float32):
            image = pipe(prompt_text, guidance_scale=8.5).images[0]
    else:
        image = pipe(prompt_text, guidance_scale=8.5).images[0]
    
    # Save and display the image
    image.save('generatedimage.png')
    img = Image.open('generatedimage.png')
    img = ImageTk.PhotoImage(img)
    lmain.configure(image=img)
    lmain.image = img  # Keep a reference to avoid garbage collection

# Create generate button
trigger = ctk.CTkButton(app, height=40, width=120, text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
