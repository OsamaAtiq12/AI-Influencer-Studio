#!/usr/bin/env python3
"""AI Influencer — person + background → still via Krea2 Identity Edit."""

from __future__ import annotations

import gradio as gr

from core.comfy_backend import nsfw_lora_ready
from core.oneshot import generate_oneshot

THEME = gr.themes.Default(
    primary_hue=gr.themes.Color(
        c50="#e6f7f3",
        c100="#b8ebe0",
        c200="#7fd4c2",
        c300="#4bb89f",
        c400="#2d9a82",
        c500="#1f7f6d",
        c600="#176557",
        c700="#124f44",
        c800="#0e3c34",
        c900="#0a2c26",
        c950="#061a17",
    ),
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("DM Sans"), "Arial", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
).set(
    body_background_fill="#0e1014",
    body_background_fill_dark="#0e1014",
    body_text_color="#e8eaed",
    body_text_color_dark="#e8eaed",
    background_fill_primary="#161a21",
    background_fill_primary_dark="#161a21",
    background_fill_secondary="#12161c",
    background_fill_secondary_dark="#12161c",
    border_color_primary="#2a303a",
    border_color_primary_dark="#2a303a",
    block_background_fill="#161a21",
    block_background_fill_dark="#161a21",
    block_border_color="#2a303a",
    block_border_color_dark="#2a303a",
    block_label_background_fill="#161a21",
    block_label_background_fill_dark="#161a21",
    block_label_text_color="#c5cad3",
    block_label_text_color_dark="#c5cad3",
    block_title_text_color="#e8eaed",
    block_title_text_color_dark="#e8eaed",
    input_background_fill="#12161c",
    input_background_fill_dark="#12161c",
    button_primary_background_fill="#2d9a82",
    button_primary_background_fill_hover="#3db090",
    button_primary_text_color="#0a1210",
    shadow_drop="none",
    shadow_drop_lg="0 12px 40px rgba(0, 0, 0, 0.45)",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

:root, .dark {
  color-scheme: dark;
}

html, body {
  background: #0e1014 !important;
  color: #e8eaed !important;
}

.gradio-container {
  background: #0e1014 !important;
  color: #e8eaed !important;
  max-width: 1120px !important;
  margin: 0 auto !important;
  padding: 1.5rem 1.25rem 2.5rem !important;
  font-family: "DM Sans", Arial, sans-serif !important;
}

footer { display: none !important; }

#ais-app {
  position: relative;
  z-index: 1;
}

#ais-app .ais-top {
  margin-bottom: 1.5rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid #2a303a;
}
#ais-app .ais-brand {
  font-size: 1.55rem;
  font-weight: 700;
  letter-spacing: -0.04em;
  margin: 0;
  color: #f2f4f7;
}
#ais-app .ais-sub {
  margin: 0.35rem 0 0;
  color: #8b939e;
  font-size: 0.95rem;
  line-height: 1.4;
  max-width: 34rem;
}
#ais-app .ais-section {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #4bb89f;
  margin: 0 0 0.55rem 0;
}
#ais-app .ais-help {
  color: #7a828e;
  font-size: 0.84rem;
  margin: 0.45rem 0 1.25rem;
  line-height: 1.45;
}

#ais-app #ais-inputs,
#ais-app #ais-result {
  position: relative;
  z-index: 1;
  min-width: 0;
}

#ais-app .ais-card {
  background: #161a21 !important;
  border: 1px solid #2a303a !important;
  border-radius: 14px !important;
}

#ais-app .ais-card img {
  max-width: 100% !important;
  object-fit: contain !important;
}

#ais-app button.primary,
#ais-app .ais-go button {
  background: #2d9a82 !important;
  color: #0a1210 !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 650 !important;
  font-size: 0.98rem !important;
  min-height: 2.85rem !important;
  box-shadow: none !important;
}
#ais-app button.primary:hover,
#ais-app .ais-go button:hover {
  background: #3db090 !important;
}

#ais-app details summary {
  font-weight: 600 !important;
  color: #e8eaed !important;
}

#ais-app #ais-inputs,
#ais-app #ais-inputs > div,
#ais-app .ais-advanced,
#ais-app .ais-advanced .block,
#ais-app .ais-advanced .form,
#ais-app .ais-advanced .slider-container {
  min-width: 0 !important;
  max-width: 100% !important;
  overflow-x: hidden !important;
}

#ais-app .ais-advanced input[type="range"] {
  width: 100% !important;
  max-width: 100% !important;
}

#ais-app .tab-nav button {
  font-weight: 600 !important;
  color: #8b939e !important;
}
#ais-app .tab-nav button.selected {
  color: #7fd4c2 !important;
  border-color: #2d9a82 !important;
}

#ais-app .how-wrap {
  max-width: 720px;
  padding: 0.5rem 0 1.5rem;
}
#ais-app .how-wrap h2 {
  margin: 0 0 0.4rem;
  font-size: 1.35rem;
  letter-spacing: -0.03em;
  color: #f2f4f7;
}
#ais-app .how-wrap .lead {
  color: #8b939e;
  font-size: 1rem;
  line-height: 1.5;
  margin: 0 0 1.75rem;
}
#ais-app .how-step {
  display: grid;
  grid-template-columns: 2.5rem 1fr;
  gap: 0.85rem;
  margin-bottom: 1.35rem;
  padding-bottom: 1.35rem;
  border-bottom: 1px solid #2a303a;
}
#ais-app .how-step:last-of-type {
  border-bottom: none;
  margin-bottom: 0.5rem;
  padding-bottom: 0;
}
#ais-app .how-num {
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 10px;
  background: #132822;
  border: 1px solid #1f4a3e;
  color: #7fd4c2;
  font-weight: 700;
  font-size: 0.85rem;
  display: flex;
  align-items: center;
  justify-content: center;
}
#ais-app .how-step h3 {
  margin: 0.15rem 0 0.35rem;
  font-size: 1.02rem;
  color: #f2f4f7;
}
#ais-app .how-step p {
  margin: 0;
  color: #9aa3ae;
  font-size: 0.92rem;
  line-height: 1.5;
}
#ais-app .how-tips {
  margin-top: 1.5rem;
  padding: 1.1rem 1.2rem;
  background: #161a21;
  border: 1px solid #2a303a;
  border-radius: 14px;
}
#ais-app .how-tips h3 {
  margin: 0 0 0.65rem;
  font-size: 0.78rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #4bb89f;
}
#ais-app .how-tips ul {
  margin: 0;
  padding-left: 1.1rem;
  color: #9aa3ae;
  font-size: 0.9rem;
  line-height: 1.55;
}
#ais-app .how-tips li { margin-bottom: 0.4rem; }
#ais-app .how-tips li:last-child { margin-bottom: 0; }
"""


def _flatten_gallery(sheets) -> list:
    if not sheets:
        return []
    items = sheets if isinstance(sheets, list) else [sheets]
    flat = []
    for item in items:
        if isinstance(item, (list, tuple)):
            flat.append(item[0])
        elif isinstance(item, dict) and "image" in item:
            flat.append(item["image"])
        elif isinstance(item, dict) and "name" in item:
            flat.append(item["name"])
        else:
            flat.append(item)
    return flat


def generate_ui(
    sheets,
    background,
    prompt: str,
    creativity: float,
    seed: int,
    steps: int,
    enable_nsfw: bool,
    nsfw_strength: float,
    progress=gr.Progress(track_tqdm=False),
):
    flat = _flatten_gallery(sheets)
    if not flat:
        raise gr.Error("Add a character sheet (at least one photo).")
    if background is None:
        raise gr.Error("Add a background photo.")
    if enable_nsfw and not nsfw_lora_ready():
        raise gr.Error("NSFW LoRA is enabled but the file is missing.")

    def cb(frac: float, msg: str) -> None:
        progress(frac, desc=msg)

    try:
        result = generate_oneshot(
            flat,
            background,
            prompt=prompt or "",
            creativity=float(creativity),
            seed=int(seed or 0),
            steps=int(steps or 12),
            enable_nsfw_lora=bool(enable_nsfw),
            nsfw_strength=float(nsfw_strength),
            progress=cb,
        )
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(str(exc)) from exc

    nsfw_note = f"NSFW on · {nsfw_strength}" if enable_nsfw else "NSFW off"
    return result.image, f"{result.path.name}  ·  {int(steps)} steps  ·  {nsfw_note}"


def _toggle_nsfw(enabled: bool):
    return gr.update(visible=bool(enabled))


def build_app() -> gr.Blocks:
    nsfw_ok = nsfw_lora_ready()

    with gr.Blocks(title="AI Influencer", fill_height=False, elem_id="ais-app") as demo:
        gr.HTML(
            """
            <div class="ais-top">
              <h1 class="ais-brand">AI Influencer</h1>
              <p class="ais-sub">Upload a character sheet and a scene, then describe what they should do in that place.</p>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("Studio"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=360, elem_id="ais-inputs"):
                        gr.HTML('<p class="ais-section">01 — Character sheet</p>')
                        sheets = gr.Gallery(
                            label="Character sheet",
                            value=None,
                            type="pil",
                            columns=2,
                            rows=2,
                            height=280,
                            object_fit="contain",
                            interactive=True,
                            show_label=True,
                            elem_classes=["ais-card"],
                        )
                        gr.HTML(
                            '<p class="ais-help">Upload front / side / back / face views, or one multi-view sheet. '
                            "A sharp face crop gives the best likeness.</p>"
                        )

                        gr.HTML('<p class="ais-section">02 — Background</p>')
                        background = gr.Image(
                            label="Scene photo",
                            type="pil",
                            height=220,
                            sources=["upload", "clipboard"],
                            show_label=True,
                            elem_classes=["ais-card"],
                        )
                        gr.HTML(
                            '<p class="ais-help">The room or location you want them inside. Keep creativity lower to preserve this layout.</p>'
                        )

                        gr.HTML('<p class="ais-section">03 — Action</p>')
                        prompt = gr.Textbox(
                            label="Describe the moment",
                            placeholder="sitting on this exact bed, laughing, looking at camera",
                            lines=3,
                            max_lines=8,
                            show_label=True,
                            elem_classes=["ais-card"],
                        )

                        with gr.Accordion("Advanced", open=False, elem_classes=["ais-advanced"]):
                            creativity = gr.Slider(
                                0.35,
                                0.85,
                                value=0.55,
                                step=0.05,
                                label="Creativity",
                                info="Lower = closer face + scene match",
                            )
                            steps = gr.Slider(
                                4,
                                28,
                                value=12,
                                step=1,
                                label="Steps",
                                info="12 is a solid Turbo default",
                            )
                            seed = gr.Number(value=0, precision=0, label="Seed")
                            enable_nsfw = gr.Checkbox(
                                label="NSFW LoRA",
                                value=False,
                                interactive=nsfw_ok,
                            )
                            nsfw_strength = gr.Slider(
                                0.5,
                                4.0,
                                value=2.0,
                                step=0.1,
                                label="NSFW strength",
                                visible=False,
                                interactive=nsfw_ok,
                            )

                        gr.HTML('<div style="height:0.75rem"></div>')
                        with gr.Row(elem_classes=["ais-go"]):
                            gen_btn = gr.Button("Generate image", variant="primary", size="lg")

                    with gr.Column(scale=6, min_width=420, elem_id="ais-result"):
                        gr.HTML('<p class="ais-section">Result</p>')
                        output = gr.Image(
                            label="Generated image",
                            type="pil",
                            height=520,
                            interactive=False,
                            show_label=True,
                            elem_classes=["ais-card"],
                        )
                        gen_info = gr.Textbox(
                            label="File",
                            lines=1,
                            interactive=False,
                            show_label=True,
                        )

            with gr.Tab("How it works"):
                gr.HTML(
                    """
                    <div class="how-wrap">
                      <h2>How it works</h2>
                      <p class="lead">
                        This studio places your person into a real scene photo using
                        Krea 2 Identity Edit — no character training required.
                      </p>

                      <div class="how-step">
                        <div class="how-num">01</div>
                        <div>
                          <h3>Character sheet</h3>
                          <p>
                            Upload clear photos of the same person: face close-up, front,
                            side, and back work best. One multi-view sheet is fine.
                            Sharp face detail locks likeness better than stylish crops.
                          </p>
                        </div>
                      </div>

                      <div class="how-step">
                        <div class="how-num">02</div>
                        <div>
                          <h3>Background</h3>
                          <p>
                            This is the room or location they should appear in.
                            The model keeps this layout as the base image, then inserts
                            your character into it.
                          </p>
                        </div>
                      </div>

                      <div class="how-step">
                        <div class="how-num">03</div>
                        <div>
                          <h3>Action</h3>
                          <p>
                            Describe pose, expression, and clothing in plain language.
                            Be specific about the scene when you want an exact match —
                            e.g. “sitting on this exact bed, looking at camera.”
                          </p>
                        </div>
                      </div>

                      <div class="how-step">
                        <div class="how-num">04</div>
                        <div>
                          <h3>Generate</h3>
                          <p>
                            ComfyUI runs Krea 2 Turbo with an Identity Edit LoRA.
                            Your scene is image 1, your person is image 2.
                            Results save locally under the outputs folder.
                          </p>
                        </div>
                      </div>

                      <div class="how-tips">
                        <h3>Tips</h3>
                        <ul>
                          <li><strong>Creativity lower</strong> keeps face and room closer to your uploads.</li>
                          <li><strong>Steps ~12</strong> is a solid Turbo default; raise only if results look unfinished.</li>
                          <li>Reference clothing can stick — say what to wear (or not wear) clearly in the prompt.</li>
                          <li>NSFW LoRA is optional under Advanced; leave it off for normal generations.</li>
                          <li>Same seed + same inputs usually reproduce a similar result.</li>
                        </ul>
                      </div>
                    </div>
                    """
                )

        enable_nsfw.change(_toggle_nsfw, inputs=[enable_nsfw], outputs=[nsfw_strength])
        gen_btn.click(
            generate_ui,
            inputs=[
                sheets,
                background,
                prompt,
                creativity,
                seed,
                steps,
                enable_nsfw,
                nsfw_strength,
            ],
            outputs=[output, gen_info],
        )
    return demo


if __name__ == "__main__":
    build_app().queue().launch(
        theme=THEME,
        css=CSS,
        favicon_path=None,
        server_name="127.0.0.1",
        server_port=7861,
    )
