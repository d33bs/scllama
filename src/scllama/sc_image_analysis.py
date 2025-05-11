# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] papermill={"duration": 0.002203, "end_time": "2025-04-24T22:42:14.689897", "exception": false, "start_time": "2025-04-24T22:42:14.687694", "status": "completed"}
# # Using `llama.cpp` vision to analyze single-cell images
#
# This notebook demonstrates how to use `llama.cpp` to analyze single-cell images.
# Note: this work assumes the data have been downloaded locally using
# `src/data/prepare_files.py`.

# + papermill={"duration": 0.441802, "end_time": "2025-04-24T22:42:15.139535", "exception": false, "start_time": "2025-04-24T22:42:14.697733", "status": "completed"}
import pathlib

import matplotlib.pyplot as plt
from skimage import io
from utils import display_response, query_llama_with_image_path

# setup a data directory reference
source_data_dir = str(pathlib.Path("../data/input").resolve())
target_data_dir = str(pathlib.Path("../data/output").resolve())

# + papermill={"duration": 0.010481, "end_time": "2025-04-24T22:42:15.151824", "exception": false, "start_time": "2025-04-24T22:42:15.141343", "status": "completed"}
# show the files
print(
    "List of files:\n",
    (file_list := list(pathlib.Path(source_data_dir).rglob("*.tif"))),
)

# + papermill={"duration": 0.187643, "end_time": "2025-04-24T22:42:15.906189", "exception": false, "start_time": "2025-04-24T22:42:15.718546", "status": "completed"}
# display images by reading them locally
for image_file in pathlib.Path(source_data_dir).rglob("*.tif"):
    plt.clf()
    plt.imshow(io.imread(image_file), cmap="gray")
    plt.axis("off")
    plt.show()
# -

# read an image into the model and ask it to describe
response = query_llama_with_image_path(
    image_path=str(file_list[0].resolve()), prompt="Describe this image."
)
# display the response as HTML rendered from markdown
display_response(response)

# read an image into the model and ask it to describe
response = query_llama_with_image_path(
    image_path=str(file_list[0].resolve()),
    # we might expect around 30 as a response
    prompt=(
        "How many objects are there in this image?"
        " Specifically, I'm looking for a count with"
        " a description of how/why and not code."
    ),
)
display_response(response)

# read an image into the model and ask it to describe
response = query_llama_with_image_path(
    image_path=str(file_list[0].resolve()),
    prompt=(
        "What is the image quality for this image"
        "(i.e. does it have good or bad quality for"
        "the domain you mentioned)?"
    ),
)
display_response(response)

# ask for features
response = query_llama_with_image_path(
    image_path=str(file_list[0].resolve()),
    prompt=(
        "Could you create me some CellProfiler features"
        " based on this image? I need to use the features"
        " to analyze the objects within from an image-based"
        " profiling perspective. I'm looking for a description"
        " of how/why and also the data itself."
    ),
)
display_response(response)
