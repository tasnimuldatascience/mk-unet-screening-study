import argparse
import csv
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    run=ROOT/args.run;out=ROOT/'output/figures';out.mkdir(parents=True,exist_ok=True)
    prov=json.loads((run/'provenance.json').read_text())
    manifest=json.loads((ROOT/prov['config']['manifest']).read_text())
    by_id={r['id']:r for r in manifest['samples']}
    history=[json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(9,3),layout='constrained')
    for ax,key,label in zip(axes,['loss','val_dice'],['Training structure loss','Validation Dice']):
        ax.plot([r['epoch'] for r in history],[r[key] for r in history],color='#087f8c');ax.set(xlabel='Epoch',ylabel=label);ax.grid(alpha=.2)
    fig.savefig(out/f'{run.name}_learning.png',dpi=160);plt.close(fig)
    rows=sorted(csv.DictReader((run/'test/per_image.csv').open()),key=lambda r:float(r['dice']))
    chosen=[rows[0],rows[len(rows)//2],rows[-1]]
    fig,axes=plt.subplots(3,3,figsize=(9,8),layout='constrained')
    for i,(case,label) in enumerate(zip(chosen,['Lowest','Median','Highest'])):
        r=by_id[case['id']]
        im=cv2.cvtColor(cv2.imread(str(ROOT/r['image'])),cv2.COLOR_BGR2RGB)
        gt=cv2.imread(str(ROOT/r['mask']),0)>20
        if r.get('foreground')=='black':gt=~gt
        pred=cv2.imread(str(run/'test/predictions'/f"{r['id']}.png"),0)>0
        for ax,img,title in zip(axes[i],[im,gt,pred],[f"{label} Dice: {float(case['dice']):.3f} | ID {r['id']}",'Ground truth','Prediction']):
            ax.imshow(img,cmap='gray',vmin=0,vmax=1 if img.ndim==2 else None);ax.set_title(title,fontsize=9);ax.axis('off')
    fig.savefig(out/f'{run.name}_cases.png',dpi=140);plt.close(fig)
    (out/f'{run.name}_cases.json').write_text(json.dumps(dict(selection='Lowest, median, highest held-out Dice; descriptive only',cases=chosen),indent=2))
