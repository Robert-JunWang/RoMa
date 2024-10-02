import os
import ssl                                        
ssl._create_default_https_context = ssl._create_unverified_context
openssl_dir, openssl_cafile = os.path.split(      
    ssl.get_default_verify_paths().openssl_cafile)
# no content in this folder
os.listdir(openssl_dir)
print(openssl_dir, openssl_cafile)
# non existent file
print(os.path.exists(os.path.join(openssl_dir, openssl_cafile)))


import torch
xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)

dinov2_weights = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", map_location="cpu")


