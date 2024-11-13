import argparse
from PIL import Image
import gradio as gr
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline

PROMPT_PREFIX = '1boy, "TOK", '
PROMPT_POSTFIX = ''
NEG_PROMPT = "nsfw, nude, lowres, (bad), error, fewer, extra, missing, worst quality, low quality, bad anatomy, unfinished, displeasing,"
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--model", type=str, default="cagliostrolab/animagine-xl-3.1"
    )
    parser.add_argument(
        "--lora", type=str, default="/hdd1/aidml/sjwi/INU_torchy_generator/train_result/outputs/animagine-xl-3.1_0505_5/animagine-xl-3.1.safetensors"
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

def predict(prompt, neg_prompt, guidance, steps, seed=1231231):
    prompt = PROMPT_PREFIX + prompt + PROMPT_POSTFIX
    results = pipe(prompt=prompt,
                   generator=torch.manual_seed(seed),
                   num_inference_steps=steps,
                   negative_prompts= neg_prompt,
                   width=1024,
                   height=1024, 
                   guidance_scale=guidance,
                   ).images[0]
    
    return results

if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda')
    #scheduler = get_my_scheduler(sample_sampler="euler_a", v_parameterization=False)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    
    pipe.load_lora_weights(args.lora)
    pipe.to(device)
    
    
    css = """
    #container{
        margin: 0 auto;
        max-width: 40rem;
    }
    #intro{
        max-width: 100%;
        text-align: center;
        margin: 0 auto;
    }
    """
    with gr.Blocks(css=css) as demo:
        with gr.Column(elem_id="container"):
            with gr.Row():
                with gr.Row():
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Insert your prompt here:", scale=5, container=False
                    )
                    prompt.value = 'swimming, sunglasses, swim tube'
            with gr.Row():
                    neg_prompt =  gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Insert your negative prompt here:", scale=5, container=False
                    )
                    neg_prompt.value = NEG_PROMPT
                    generate_bt = gr.Button("Generate", scale=1)

            image = gr.Image(type="filepath")
            with gr.Accordion("Advanced options", open=False):
                guidance = gr.Slider(
                    label="Guidance", minimum=0.0, maximum=10, value=7, step=0.001
                )
                steps = gr.Slider(label="Steps", value=30, minimum=2, maximum=40, step=1)
                seed = gr.Slider(
                    randomize=True, minimum=0, maximum=12013012031030, label="Seed", step=1
                )


            inputs = [prompt, neg_prompt, guidance, steps, seed]
            generate_bt.click(fn=predict, inputs=inputs, outputs=image)


    demo.queue(api_open=False)
    demo.launch(show_api=False, share=True)
