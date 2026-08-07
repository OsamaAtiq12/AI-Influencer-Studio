# AI Influencer Studio

Create AI influencer images on your own computer.

You upload a **character sheet** (photos of the person) and a **background** (the place you want them in). The app places that person into the scene using **Krea 2 Identity Edit** — **no character LoRA training required**.

---

## What you need (minimum specs)

This app runs locally on your NVIDIA graphics card. It will not run well on a laptop without a proper NVIDIA GPU.

| Item | Minimum | Recommended |
|------|---------|-------------|
| **GPU** | NVIDIA with **12 GB VRAM** (e.g. RTX 3060 12GB, RTX 4070, RTX 5070) | 12 GB or more |
| **System RAM** | **32 GB** | 32 GB+ |
| **Storage** | **~40 GB free** on an SSD | SSD preferred |
| **OS** | Windows 10 or 11 (64-bit) | Windows 11 |
| **Python** | 3.10 – 3.12 | 3.12 |
| **Internet** | Needed once to download models | — |

**Notes for beginners**

- An **8 GB** GPU is usually too small for this setup.
- You do **not** need to train a model on someone’s face.
- Everything stays on your PC (no paid cloud API required for the main workflow).

---

## What this app does

1. You add photos of a person (**character sheet** — face / front / side works best).
2. You add a **scene photo** (bedroom, street, studio, etc.).
3. You type a short **action** (pose, expression, clothing).
4. Click **Generate** — ComfyUI runs Krea 2 Turbo + Identity Edit and saves the image locally.

Optional: NSFW LoRA can be enabled in Advanced settings. Use responsibly and follow platform rules if you post results.

---

## Installation (simple path)

### 1. Install Python

1. Download Python **3.12** from [python.org](https://www.python.org/downloads/).
2. During install, check **“Add Python to PATH”**.
3. Open **PowerShell** or **Command Prompt**.

### 2. Download this project

```bash
git clone https://github.com/OsamaAtiq12/AI-Influencer-Studio.git
cd AI-Influencer-Studio
```

(Or download the ZIP from GitHub and unzip it, then open a terminal in that folder.)

### 3. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` at the start of your terminal line.

### 4. Install PyTorch (GPU version)

Install CUDA PyTorch for your NVIDIA driver. Example for CUDA 12.8:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

If you are unsure which CUDA build to use, check [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### 5. Install the rest of the packages

```bash
pip install -r requirements.txt
```

### 6. Install ComfyUI

```bash
mkdir vendor
git clone https://github.com/comfyanonymous/ComfyUI.git vendor\ComfyUI-tmp
```

Then install ComfyUI’s Python dependencies into the same venv (from the ComfyUI folder), following ComfyUI’s own install notes if needed.

### 7. Download the AI models (~18 GB)

```bash
python scripts/download_comfy_fp8.py
```

This downloads:

- Krea 2 Turbo FP8
- Qwen text encoder
- VAE

### 8. Add the Identity Edit LoRA

Place the Krea 2 Identity Edit LoRA here:

```text
checkpoints/comfy/loras/Krea2/krea2_identity_edit_v1_2.safetensors
```

(or `krea2_identity_edit_v1_2_r64.safetensors`)

Without this file, identity locking will not work.

### 9. (Optional) NSFW LoRA

If you use it, put it here:

```text
checkpoints/comfy/loras/Krea2/krea2filterbypass3.safetensors
```

---

## How to run

You need **two** programs running.

### Terminal 1 — ComfyUI (image engine)

```bash
.venv\Scripts\activate
python scripts/run_comfyui.py
```

Or:

```bash
cd vendor\ComfyUI-tmp
..\..\.venv\Scripts\python.exe -u main.py --listen 127.0.0.1 --port 8188
```

Leave this window open. Open [http://127.0.0.1:8188](http://127.0.0.1:8188) to confirm it started.

### Terminal 2 — Studio (the website UI)

```bash
.venv\Scripts\activate
python app.py
```

Open the studio in your browser:

- [http://127.0.0.1:7861](http://127.0.0.1:7861)

(If that port is busy, check the terminal — it will print the correct link.)

---

## How to use the Studio (1 minute)

1. Open the **Studio** tab.
2. **Character sheet** — upload clear photos of the same person (face close-up helps a lot).
3. **Background** — upload the room or location photo.
4. **Action** — describe the moment, e.g. `sitting on this exact bed, looking at camera`.
5. Click **Generate image**.
6. Find the result in the Result panel (files also save under `outputs/`).

**Tips**

- Lower **Creativity** = closer match to the face and room.
- **Steps ~12** is a good default for Turbo.
- Say clothing clearly in the prompt if the sheet’s clothes keep showing up.
- Use a sharp face crop for better likeness.

The **How it works** tab in the app explains the same flow.

---

## Project folders (what matters)

```text
app.py                 → Studio website (Gradio)
core/                  → Generation + ComfyUI connection
scripts/               → Setup / download / run helpers
checkpoints/comfy/     → Downloaded models (not in git — you download these)
vendor/ComfyUI-tmp/    → Local ComfyUI (not in git — you clone this)
outputs/               → Generated images
```

Large model files are **not** included in this repository (they are too big). You download them after install.

---

## Common problems

| Problem | What to try |
|--------|-------------|
| Studio page won’t open | Make sure Terminal 2 is running `python app.py` |
| Generate fails / connection error | Make sure ComfyUI is running on port **8188** |
| Out of memory / CUDA error | Close other GPU apps; use 12 GB+ VRAM; keep resolution reasonable |
| Face doesn’t look like the person | Add a sharper face photo; lower Creativity; use more angles in the sheet |
| Missing model error | Re-run `python scripts/download_comfy_fp8.py` and check the Identity Edit LoRA path |

---

## License notes

- **This app’s code** — use under the license you choose for this repo.
- **Krea 2 model weights** — follow the [Krea 2 Community License](https://www.krea.ai/krea-2-licensing). Commercial use has revenue limits; read Krea’s terms before selling products based on the model.

---

## Links

- [Krea 2](https://github.com/krea-ai/krea-2)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Krea 2 licensing](https://www.krea.ai/krea-2-licensing)
