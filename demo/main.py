from js import document, console, Uint8Array, window, File, Object
from pyodide.ffi import create_proxy
from pyodide.http import pyfetch
import asyncio
import io
from PIL import Image, ImageFilter
import numpy as np
from js import ort
from js.ort import Tensor as OrtTensor

from pyodide.ffi import to_js

from PIL import Image
from PIL.Image import Resampling


async def inference(input_image):
    session_link = 'https://github.com/gosha20777/py-script-test/releases/download/0.1.0/fiveK.onnx'
    #session_link = 'http://0.0.0.0:8000/fiveK.onnx'
    session = await ort.InferenceSession.create(session_link)
    input_img = input_image.resize([720,480], Resampling.BILINEAR)
    input_img = np.asarray(input_img, dtype=np.float32) / 255.0
    input_img = np.expand_dims(input_img, 0)
    input_img = np.transpose(input_img, (0,3,1,2))
    print("Shape: ", input_img.shape)
    input_img = input_img.reshape(1*3*480*720) 
    
    js_arr = to_js(input_img)
    img_tensor = OrtTensor.new("float32", js_arr, to_js([1, 3, 480, 720]))
    feeds = to_js({'onnx::ReduceMean_0': img_tensor})
    console.log("feeds: ", feeds)
    results = await session.run(Object.fromEntries(feeds))
    results = results.to_py()['1103']
    console.log("Results: ", results)


    output_img = np.asarray(results.data.to_py(), dtype=np.float32).reshape(1, 3, 480, 720)
    print("Output Shape: ", output_img.shape)
    output_img = np.transpose(output_img, (0,2,3,1))[0]
    print("Output Shape: ", output_img.shape)
    output_img = Image.fromarray(np.uint8((output_img * 255.0).clip(0, 255)), mode='RGB')
    return output_img


async def process_image(input_image):
    input_stream = io.BytesIO()
    input_image.save(input_stream, format="PNG")
    input_image_file = File.new([Uint8Array.new(input_stream.getvalue())], "input_img.png", {type: "image/png"})
    document.getElementById("input-image").src = window.URL.createObjectURL(input_image_file)

    output_image = await inference(input_image)
    
    output_stream = io.BytesIO()
    output_image.save(output_stream, format="PNG")
    output_image_file = File.new([Uint8Array.new(output_stream.getvalue())], "output_img.png", {type: "image/png"})
    document.getElementById("output-image").src = window.URL.createObjectURL(output_image_file)


async def _upload_and_show(e):
    #Get the first file from upload
    file_list = e.target.files
    first_item = file_list.item(0)

    #Get the data from the files arrayBuffer as an array of unsigned bytes
    array_buf = Uint8Array.new(await first_item.arrayBuffer())

    #BytesIO wants a bytes-like object, so convert to bytearray first
    bytes_list = bytearray(array_buf)
    my_bytes = io.BytesIO(bytes_list) 

    #Create PIL image from np array
    my_image = Image.open(my_bytes)

    #Log some of the image data for testing
    console.log(f"{my_image.format= } {my_image.width= } {my_image.height= }")

    await process_image(my_image)
    

async def _select_and_show(e):
    task_name = document.getElementById("task-select").value
    img_mane = e.target.value
    url = 'https://cataas.com/cat'

    if task_name == 'rgb-to-rgb':
        if img_mane == '1':
            url = 'https://raw.githubusercontent.com/gosha20777/cmKAN/refs/heads/main/data/samples/rgb2rgb/input/01.jpg'

    response = await pyfetch(url, method="GET")
    filename = 'select.png'
    with open(filename, mode="wb") as file:
        file.write(await response.bytes())

    #Create PIL image from np array
    my_image = Image.open(filename)

    #Log some of the image data for testing
    console.log(f"{my_image.format= } {my_image.width= } {my_image.height= }")

    await process_image(my_image)


async def _init_and_show():
    response = await pyfetch('https://cataas.com/cat', method="GET")
    filename = 'select.png'
    with open(filename, mode="wb") as file:
        file.write(await response.bytes())

    #Create PIL image from np array
    my_image = Image.open(filename)

    #Log some of the image data for testing
    console.log(f"{my_image.format= } {my_image.width= } {my_image.height= }")

    await process_image(my_image)


async def main():
    # Run image processing code above whenever file is uploaded    
    upload_file = create_proxy(_upload_and_show)
    document.getElementById("file-input").addEventListener("change", upload_file)

    upload_file = create_proxy(_select_and_show)
    document.getElementById("image-select").addEventListener("change", upload_file)

    await _init_and_show()

asyncio.ensure_future(main())
