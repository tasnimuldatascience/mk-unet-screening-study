import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from .model import build_model
from .run import output

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True); p.add_argument('--image',required=True); p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args(); torch.set_num_threads(4)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    ck=torch.load(a.checkpoint,map_location=device,weights_only=False)
    model=build_model().to(device);model.load_state_dict(ck['model']);model.eval()
    im=cv2.imread(a.image)
    if im is None: raise ValueError(f'Cannot read {a.image}')
    rgb=cv2.cvtColor(cv2.resize(im,(ck['config']['size'],)*2),cv2.COLOR_BGR2RGB).astype('float32')/255
    x=(rgb-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    with torch.inference_mode():
        prob=F.interpolate(output(model,torch.from_numpy(x.transpose(2,0,1).copy())[None].to(device)),im.shape[:2],mode='bilinear',align_corners=False).sigmoid()[0,0].cpu().numpy()
    norm=(prob-prob.min())/(prob.max()-prob.min()+1e-8)
    mask=norm>=.5
    dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    np.save(dest/'probability.npy',prob)
    cv2.imwrite(str(dest/'mask.png'),mask.astype('uint8')*255)
    cv2.imwrite(str(dest/'mask_fixed_threshold.png'),(prob>=.5).astype('uint8')*255)
    overlay=im.copy();overlay[mask]=(im[mask]*.5+np.array([40,200,40])*.5).astype('uint8')
    cv2.imwrite(str(dest/'overlay.png'),overlay)
    (dest/'metadata.json').write_text(json.dumps(dict(image=a.image,checkpoint=a.checkpoint,shape=list(im.shape[:2]),normalization='per-image min-max, threshold 0.5'),indent=2))
    print(dest)

if __name__=='__main__':main()
