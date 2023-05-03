# %%
import os, sys
import torch
import pandas as pd
import tqdm

# @markdown Load dataset: `Hatefulmemes_train`
from PIL import Image
from transformers import FlavaProcessor, FlavaModel

model = FlavaModel.from_pretrained("facebook/flava-full").to('cuda:0')
processor = FlavaProcessor.from_pretrained("facebook/flava-full")

def encoding_with_flava(data_path, meta_data, image_path_renaming, max_length=512, phases=['train', 'dev', 'test']):
    for phase in phases:
        error_cases = []
        for ind in tqdm.tqdm(meta_data[phase].index):
            image_path, text = meta_data[phase].loc[ind][['img', 'text']]
            save_name = image_path_renaming(image_path)

            image = Image.open(os.path.join(data_path, image_path))
            # try:
            inputs = processor(
                text=[text], 
                images=[image], 
                return_tensors="pt", padding=True, max_length=max_length, truncation=True,
                return_codebook_pixels=False,
            )
            inputs = {k: v.to('cuda:0') for k, v in inputs.items()}

            outputs = model(**inputs)
            image_embeddings = outputs.image_embeddings.data.squeeze() # Batch size X (Number of image patches + 1) x Hidden size => 2 X 197 X 768
            text_embeddings = outputs.text_embeddings.data.squeeze() # Batch size X (Text sequence length + 1) X Hidden size => 2 X 77 X 768

            path = f"{data_path}/flava_embeds_{max_length}/{'/'.join(save_name.split('/')[:-1])}"    
            if not os.path.isdir(path):
                os.makedirs(path)
            torch.save(image_embeddings.data.squeeze(), 
                        f"{data_path}/flava_embeds_{max_length}/{save_name}.img")
            torch.save(text_embeddings.data.squeeze(), 
                        f"{data_path}/flava_embeds_{max_length}/{save_name}.text")

        print(phase, len(error_cases))
        with open(f"{data_path}/flava_embeds_{max_length}/{phase}_unseen_error_cases.txt", 'w') as f:
            for ind in error_cases:
                f.write(f"{ind}\n")

def generation_for_hatefulmeme():
    data_path = '/gpfs/data/geraslab/Nan/multi_modal/hateful-meme-dataset'
    train_meta_data = pd.read_json(path_or_buf=os.path.join(data_path, 'train.jsonl'), lines=True)
    dev_meta_data = pd.read_json(path_or_buf=os.path.join(data_path, 'dev_unseen.jsonl'), lines=True)
    test_meta_data = pd.read_json(path_or_buf=os.path.join(data_path, 'test_unseen.jsonl'), lines=True)

    meta_data = {'train': train_meta_data, 
                'dev': dev_meta_data,
                'test': test_meta_data}
    
    image_path_renaming = lambda image_path: image_path.split('/')[-1].split('.')[0]
    
    os.makedirs(f"{data_path}/flava_embeds", exist_ok=True)
    encoding_with_flava(data_path, meta_data, image_path_renaming, phases=['train', 'dev', 'test'])

def generation_for_food101(max_length):
    data_path = '/gpfs/data/geraslab/Nan/multi_modal/food101'
    train_meta_data = pd.read_json(path_or_buf=os.path.join(data_path, 'train.jsonl'), lines=True)
    dev_meta_data = pd.read_json(path_or_buf=os.path.join(data_path, 'dev.jsonl'), lines=True)
    test_meta_data = pd.read_json(path_or_buf=os.path.join(data_path, 'test.jsonl'), lines=True)

    meta_data = {'train': train_meta_data, 
                'dev': dev_meta_data,
                'test': test_meta_data}
    
    image_path_renaming = lambda image_path: image_path.split('.')[0]    
    os.makedirs(f"{data_path}/flava_embeds_{max_length}", exist_ok=True)
    encoding_with_flava(data_path, meta_data, image_path_renaming, max_length, phases=['train', 'dev', 'test'])


if __name__ == "__main__":
    # sys.argv[1] = max_length
    print(sys.argv[1])
    generation_for_food101(int(sys.argv[1]))


# # %%
# phase='train'
# ind = 1753
# image_path, text = meta_data[phase].loc[ind][['img', 'text']]
# save_name = image_path.split('/')[-1].split('.')[0]

# image = Image.open(os.path.join(data_path, image_path))
# plt.imshow(image)

# # %%
# inputs = processor(
#     text=[text], 
#     images=[image], 
#     return_tensors="pt", padding="max_length", max_length=77,
#     return_codebook_pixels=False,
# )
