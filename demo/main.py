from js import document, console, Uint8Array, window, File
from pyodide.ffi import create_proxy
from pyodide.http import pyfetch
import asyncio
import io
from PIL import Image, ImageFilter


async def inference(input_image):
    output_image = input_image.convert('L')
    return output_image


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
    #Get the first file from upload
    img_mane = e.target.value
    response = await pyfetch('https://cataas.com/cat', method="GET")
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
