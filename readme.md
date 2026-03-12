<h2 align="center">
    <br/>
    SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation<br>
</h2>

## Environment Setup

```bash
conda create -n sa python=3.11
source install_env.sh
```

## Run SceneAssistant

Modify `llm.py`, including the "base_url" parameter. By default, we use zhizengzeng, which is a relay platform. You can modify `llm.py` to use a platform you have access to.

Edit the prompt list in `run.sh` and keep only the scene prompt you want to generate, then run:
```bash
source run.sh
```
This will generate the scene.

To reduce GPU memory usage, you can modify `threedgen.py` and `image_gen.py` so they do not load all models at startup.

For resume and human-editing workflows, refer to the comments in `run.sh`.

## Render Generated Scenes

We use Blender EEVEE as the renderer.

The core rendering script is `blender_render.py`. The `render_from_json.py` script supports rendering intermediate steps from the agent workflow and can also export Blender files.

See the script itself and `visualize.sh` for more details.

## Output Format

We provide `outputs/a_laundromat_section_with_two_front_loading_washin_20260228_183615` as a reference output.

You can view the folder contents here:
https://drive.google.com/drive/folders/1TW6E9Nkd35YV3zPW_QqOIm0SO9vzwUD5?usp=sharing

There are 20 steps in total, with `i` ranging from 1 to 20.

The state before step `i` equals the state after step `i-1`. `step_i.png` and `step_i_state.json` represent the state before step `i`. `scene.json` represents the final generated scene.

## Acknowledgements

Our code is building upon these projects. We sincerely thank the authors for open-sourcing their awesome projects.

https://github.com/Tencent-Hunyuan/Hunyuan3D-2

https://github.com/Tongyi-MAI/Z-Image