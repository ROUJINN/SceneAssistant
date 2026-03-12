# Install core dependencies.
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/huggingface/diffusers
pip install git+https://github.com/openai/CLIP.git
pip install tyro google-genai openai fal-client trimesh pillow transformers fast-simplification sentencepiece icecream

# Install Hunyuan3D-2 (see the official repository for details).
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
cd Hunyuan3D-2
pip install -r requirements.txt 
pip install -e . --config-settings editable_mode=compat
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd ../../..
cd hy3dgen/texgen/differentiable_renderer
python3 setup.py install

# Install bpy.
pip install bpy==4.1.0 --extra-index-url https://download.blender.org/pypi/
pip install mathutils tqdm kiui[full]
