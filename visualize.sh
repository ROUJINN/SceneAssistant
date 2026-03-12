folder=$1

python convert_chat_to_readable.py --dir $folder

python visualize_steps.py --dir $folder

python render_from_json.py --mode agent --json_path $folder/scene.json --output $folder/render_agent.png 

python render_from_json.py --mode final --json_path $folder/scene.json --output $folder/render_final.png

# python render_from_json.py --mode final --json_path $folder/scene.json --output $folder/render_final.png --save_blend