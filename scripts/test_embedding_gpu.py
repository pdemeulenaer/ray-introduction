import test_asset_image_embedding
from sentence_transformers import SentenceTransformer
from PIL import Image
import os
import torch
import time
import ray

ray.init('localhost:6379') #(runtime_env={"working_dir": "/home/philippe/Documents/Northell/repos/itg-poc/test/"}) #(num_cpus=4) # when connecting to existing cluster, num_cpus must not be provided

# https://github.com/ray-project/ray/issues/33798
# device = torch.device('cuda' if torch.cuda.
#                       is_available() else 'cpu')
# localGpu_num = torch.cuda.device_count()
# localGpu_str = str(list(range(localGpu_num))).strip('[]')
# os.environ['CUDA_VISIBLE_DEVICES']=localGpu_str
# print("CUDA_VISIBLE_DEVICES",os.getenv("CUDA_VISIBLE_DEVICES"))
# ray.init(num_cpus=cpu_count(), num_gpus=1)
# print('ray.init()',ray.get_gpu_ids())

# os.environ["CUDA_VISIBLE_DEVICES"] ="0"
# print(torch.cuda.current_device())

# # @ray.remote(num_gpus=0.5)
# @ray.remote(num_cpus=4, num_gpus=1)
# def get_embeddings_local(img_model,images):

#     if isinstance(images, list):
#         if isinstance(images[0], Image.Image):
#             vectors = img_model.encode(images)
#         else:
#             images_pil = [Image.fromarray(bytes_to_array(image)) for image in images]
#             vectors = img_model.encode(images_pil)
#         vectors = [vector.tolist() for vector in vectors]
#     else:
#         vectors = img_model.encode(images)
#         vectors = [vectors.tolist()]
#     return vectors

images, video_info = test_asset_image_embedding.get_frames_local('HLSY0012000H.mp4', sampling_rate=2)
print(type(images[0]))

img_model = SentenceTransformer('clip-ViT-B-32', device='cuda')
# img_model = SentenceTransformer('clip-ViT-B-32')

# Put the model in the shared object store.
ref_img_model = ray.put(img_model)

# vectors = get_embeddings_local(images)

start = time.time()
x_vectors = test_asset_image_embedding.get_embeddings_local.remote(ref_img_model,images)
vectors = ray.get([x_vectors])

# x_vectors = test_asset_image_embedding.get_embeddings_local.remote(ref_img_model,images*10)
# vectors = ray.get([x_vectors])
end = time.time()

print('time elapsed: ', end-start)

print(vectors[0][0])