import io
import os
import tempfile
import urllib
from io import BytesIO
from typing import Optional

import cv2
import ffmpeg
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer

import ray

# ray.init()

# img_model = SentenceTransformer('clip-ViT-B-32')
# ref_img_model = ray.put(img_model)


def get_scaled_size(width, height):
    target_width = 224
    w_percent = (target_width / float(width))
    h_size = int((float(height) * float(w_percent)))
    return target_width, h_size


def array_to_bytes(x: np.ndarray) -> bytes:
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def get_frames_test(content, sampling_rate=10, resize=False):
    with tempfile.NamedTemporaryFile() as f:
        f.write(io.BytesIO(content).getbuffer())

        probe = ffmpeg.probe(f.name, threads=1)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        if resize:
            width, height = get_scaled_size(int(video_info['width']), int(video_info['height']))
        else:
            width = int(video_info['width'])
            height = int(video_info['height'])

        out, err = (
            ffmpeg
                .input(f.name, threads=1)
                .filter('scale', width, height)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
        )
        frames = (
            np
                .frombuffer(out, np.uint8)
                .reshape([-1, height, width, 3])
        )
        f.close()

    if sampling_rate > 0:
        images = [array_to_bytes(frame) for frame in frames[::sampling_rate, :]]
    else:
        images = [array_to_bytes(frame) for frame in frames]
    video_info['frames_per_sec'] = eval(video_info['avg_frame_rate'])
    return images, video_info


def get_frames_cv2(in_filename, sampling_rate=10):
    vid_capture = cv2.VideoCapture(in_filename)
    frames = []

    while (vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if ret == True:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(img))
        else:
            break

    height = vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    ext = os.path.splitext(in_filename)[-1]
    framecount = vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frames_per_sec = vid_capture.get(cv2.CAP_PROP_FPS)
    video_info = {'duration': framecount / frames_per_sec if frames_per_sec > 0 else None,
                  'width': width,
                  'height': height,
                  'display_aspect_ratio': f'{width}x{height}',
                  'format': ext,
                  'nb_frames': framecount,
                  'frames_per_sec': frames_per_sec
                  }

    if sampling_rate > 0:
        images = [frame for frame in frames[::sampling_rate]]
    else:
        images = frames

    vid_capture.release()
    return images, video_info


def get_frames_movie(in_filename, sampling_rate=10):
    video = VideoFileClip(in_filename)
    frames = []
    for i, frame in video.iter_frames(with_times=True):
        # Convert the frame (which is a numpy array) to an image
        frames.append(Image.fromarray(frame))
    video_info = {attr: getattr(video, attr) for attr in dir(video) if not attr.startswith("__") and not callable(
        getattr(video, attr)) and attr != 'audio' and attr != 'reader'}
    video.close()
    del video

    if sampling_rate > 0:
        images = [frame for frame in frames[::sampling_rate]]
    else:
        images = frames

    video_info['frames_per_sec'] = video_info['fps']
    video_info['width'] = int(video_info['w'])
    video_info['height'] = int(video_info['h'])
    video_info['nb_frames'] = len(frames)

    return images, video_info


def get_frames_local(in_filename, sampling_rate=10, method='ffmpeg'):
    # if in_filename.find('dbfs') > 0:
    #     filename = in_filename
    # else:
    #     filename = f'/dbfs/{in_filename}'
    filename = in_filename

    if method == 'cv2':
        images, video_info = get_frames_cv2(in_filename=in_filename, sampling_rate=sampling_rate)
    elif method == 'movie':
        images, video_info = get_frames_movie(in_filename=in_filename, sampling_rate=sampling_rate)
    else:
        with open(filename, 'rb') as file:
            content = file.read()
            images, video_info = get_frames_test(content, sampling_rate)

    return images, video_info


def download_asset_from_url(url: str, asset_uuid: str, bearer: Optional[str] = None) -> str:
    """
    bearer for itg: MoRUpyaEmENbYZDiirqY8Q0vJjFfurCE
    """
    path_name = './data/.cache/'

    if not os.path.isdir(path_name):
        os.mkdir(path_name)

    filename = f'{path_name}test_{asset_uuid}'

    if bearer is not None:
        opener = urllib.request.build_opener()
        opener.addheaders = [('Authorization', f'Bearer {bearer}')]
        urllib.request.install_opener(opener)

    urllib.request.urlretrieve(url, filename)

    return filename

# @ray.remote(num_gpus=0.5)
@ray.remote(num_cpus=4, num_gpus=1)
def get_embeddings_local(img_model,images):

    if isinstance(images, list):
        if isinstance(images[0], Image.Image):
            vectors = img_model.encode(images)
        else:
            images_pil = [Image.fromarray(bytes_to_array(image)) for image in images]
            vectors = img_model.encode(images_pil)
        vectors = [vector.tolist() for vector in vectors]
    else:
        vectors = img_model.encode(images)
        vectors = [vectors.tolist()]
    return vectors


def upload_embeddings(df, collection_name='ray_test'):
    """
    Uploads df to Milvus index

    Parameters
    ----------
    df: Data Frame
        with columns = ['uuid', 'frame_number', 'filename', 'embedding', 'processor'], processor = ['ray', 'azure']
    collection_name: str
        Name of the collection

    Returns
    -------
    ins_resp: Dict
        Milvus api response
    """
    milvus_uri = "https://in01-2d3d794ba1bc66f.aws-eu-central-1.vectordb.zillizcloud.com:19541"
    user = "db_admin"
    password = "Dx5$Apb&-DNF5uhn"
    connections.connect("default", uri=milvus_uri, user=user, password=password)

    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)

        entities = {
            "uuid": df['uuid'].values,
            "frame_number": df["frame_number"].values,
            "filename": df["filename"].values,
            "embedding": df["embedding"].values,
            "processor": df['processor'].values
        }

        ins_resp = collection.insert(entities)
    else:
        ins_resp = None

    return ins_resp
