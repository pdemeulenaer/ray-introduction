import test_asset_image_embedding
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PIL import Image
from typing import Dict
import numpy as np
import os
import torch
import time
import ray
import cv2

# ray.init('localhost:6379') #(runtime_env={"working_dir": "/home/philippe/Documents/Northell/repos/itg-poc/test/"}) #(num_cpus=4) # when connecting to existing cluster, num_cpus must not be provided

# images, video_info = test_asset_image_embedding.get_frames_local('HLSY0012000H.mp4')
# print(type(images[0]))

# ds = ray.data.read_images("./images/", mode="RGB")
# print(ds.schema())

# single_batch = ds.take_batch(10)
# # print(single_batch)


# Get the frames
images, video_info = test_asset_image_embedding.get_frames_cv2(in_filename='HLSY0012000H.mp4', sampling_rate=2)
print(type(images[0]), len(images))

# Transform into numpy arrays
numpy_array_list = [np.asarray(image) for image in images]
numpy_array = np.stack(numpy_array_list, axis=0)
print(numpy_array.shape)
ds = ray.data.from_numpy(numpy_array)
ds = ds.repartition(2,shuffle=False)
print(ds.schema())
print(ds.materialize())

# DUMMY IMAGES
# numpy_array = np.random.rand(200, 1080, 1920, 3)
# numpy_array = np.random.rand(2000, 224, 224, 3)
# print(numpy_array.shape)
# ds = ray.data.from_numpy(numpy_array)
# ds = ds.repartition(2,shuffle=False)
# print(ds.schema())
# print(ds.materialize())


# Pick the largest batch size that can fit on our GPUs
BATCH_SIZE = 50 #1024



class EmbeddingClass:
    def __init__(self): 
        # If doing CPU inference, set `device="cpu"` instead.
        self.model = SentenceTransformer('clip-ViT-B-32', device='cuda')

    def __call__(self, batch: Dict[str, np.ndarray]):
        # Convert the numpy array of images into a list of PIL images which is the format the HF pipeline expects.
        outputs = self.model.encode(
            [Image.fromarray(image_array.astype(np.uint8)) for image_array in batch["data"]], 
            # top_k=1, 
            batch_size=BATCH_SIZE)
 
        vectors = [outputs.tolist()]
        return {"embedding": outputs}       
    
predictions = ds.map_batches(
    EmbeddingClass, # ImageClassifier,
    concurrency=2, # number of workers to use
    num_gpus=1,  #1 #Specify 1 GPU per model replica.
    # num_cpus=2,  # Specify N CPUs per model replica.
    batch_size=BATCH_SIZE # Use the largest batch size that can fit on our GPUs
)    

start = time.time()
# # Let’s take a small batch and verify the results.
for i in range(0,1):
    prediction_batch = predictions.take_batch(2000)
# prediction_batch = predictions.take_batch(300)
print(prediction_batch)
print(prediction_batch['embedding'].shape)
end = time.time()

print('time elapsed: ', end-start)





# RUBBISH AFTER THIS


# # Create a NumPy array for demonstration
# numpy_array = np.random.rand(10, 224, 224, 3)

# # Convert NumPy array to Ray dataset
# ray_dataset = ray.data.from_numpy(numpy_array)

# # Create an instance of your EmbeddingClass
# embedding_class = EmbeddingClass()

# # Apply the map_batches operation
# result_dataset = ray_dataset.map_batches(embedding_class)

# prediction_batch = result_dataset.take_batch(5)
# print(prediction_batch)
# print(prediction_batch['embedding'].shape)

# # Accessing the result dataset
# for item in result_dataset:
#     print(item)


# images, video_info = test_asset_image_embedding.get_frames_local('HLSY0012000H.mp4')
# print(type(images[0]), len(images))

# img_model = SentenceTransformer('clip-ViT-B-32')

# img_model = SentenceTransformer('clip-ViT-B-32', device='cuda')
# # img_model = SentenceTransformer('clip-ViT-B-32')

# # Put the model in the shared object store.
# ref_img_model = ray.put(img_model)

# # vectors = get_embeddings_local(images)

# start = time.time()
# x_vectors = test_asset_image_embedding.get_embeddings_local.remote(ref_img_model,images*10)
# vectors = ray.get([x_vectors])

# x_vectors = test_asset_image_embedding.get_embeddings_local.remote(ref_img_model,images*10)
# vectors = ray.get([x_vectors])
# end = time.time()

# print('time elapsed: ', end-start)

# print(vectors[0][0])





# # Note, you must have GPUs on your head node in order to do this with GPUs.
# # If doing CPU inference, set device="cpu" instead.
# classifier = pipeline("image-classification", model="google/vit-base-patch16-224")#, device="cuda:0")
# outputs = classifier([Image.fromarray(image_array) for image_array in single_batch["image"]], top_k=1, batch_size=10)
# del classifier # Delete the classifier to free up GPU memory.
# print(outputs)


# class ImageClassifier:
#     def __init__(self):
#         # If doing CPU inference, set `device="cpu"` instead.
#         self.classifier = pipeline("image-classification", model="google/vit-base-patch16-224", device="cpu") #, device="cuda:0")

#     def __call__(self, batch: Dict[str, np.ndarray]):
#         # Convert the numpy array of images into a list of PIL images which is the format the HF pipeline expects.
#         outputs = self.classifier(
#             [Image.fromarray(image_array) for image_array in batch["image"]], 
#             top_k=1, 
#             batch_size=BATCH_SIZE)
        
#         # `outputs` is a list of length-one lists. For example:
#         # [[{'score': '...', 'label': '...'}], ..., [{'score': '...', 'label': '...'}]]
#         batch["score"] = [output[0]["score"] for output in outputs]
#         batch["label"] = [output[0]["label"] for output in outputs]
#         return batch