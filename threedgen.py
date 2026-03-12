import time

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from PIL import Image

model_path = "tencent/Hunyuan3D-2"
pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)
rembg = BackgroundRemover()


def rmbg(image):

    image = rembg(image)
    return image


def mesh_gen(image):

    time1 = time.time()
    mesh = pipeline_shapegen(image=image)[0]
    print(f"Original mesh face count: {len(mesh.faces)}")
    print(f"Original mesh vertex count: {len(mesh.vertices)}")
    simplified_mesh = mesh.simplify_quadric_decimation(
        face_count=min(50000, len(mesh.faces))
    )  # Guard against errors when the original mesh has very few faces.
    print(f"\nSimplified mesh face count: {len(simplified_mesh.faces)}")
    print(f"Simplified mesh vertex count: {len(simplified_mesh.vertices)}")
    time2 = time.time()
    print(f"Shape generation time: {time2 - time1:.2f}s")
    # del pipeline_shapegen
    time1 = time.time()
    # pipeline_texgen.enable_model_cpu_offload()
    mesh = pipeline_texgen(simplified_mesh, image=image)
    time2 = time.time()
    print(f"Texture generation time: {time2 - time1:.2f}s")
    # del pipeline_texgen
    return mesh


if __name__ == "__main__":
    img = Image.open("images/test-bed-0.png")
    img = rmbg(img)
    img.save("images/test-bed-0-rmbg.png")
    mesh = mesh_gen("images/test-bed-0-rmbg.png")
    mesh.export("assets/bed.glb")
