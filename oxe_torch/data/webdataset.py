import h5py
import torch
import webdataset as wds
from webdataset import writer
import numpy as np
from PIL import Image
import io
from io import BytesIO

def write_dict_with_image_and_text():
    # Assume we have a list of images and texts
    images = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    texts = ['text1', 'text2', 'text3']

    with writer.TarWriter('output.tar') as sink:
        for image, text in zip(images, texts):
            # Open the image and convert it to a PyTorch tensor
            with Image.open(image) as img:
                tensor = torch.from_numpy(np.array(img))
            # Write the image and text to the WebDataset
            sink.write({
                'image.jpg': tensor,
                'text.txt': text
            })

def process_group(group, sink):
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            tensor = torch.from_numpy(np.array(item))
            key = key + '.pth' if isinstance(tensor, torch.Tensor) else key
            sink.write({key: tensor})
        elif isinstance(item, h5py.Group):
            process_group(item, sink)

def tensor_to_bytes(tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()

def from_hdf5(input_file, output_file):
    with h5py.File(input_file, 'r') as f, writer.TarWriter(output_file) as sink:
        handlers = {'entry1': lambda x: x, 'entry2': lambda x: x}
        sink
        sink.write(f)
        # process_group(f, sink)


if __name__ == "__main__":
    write_dict_with_image_and_text()