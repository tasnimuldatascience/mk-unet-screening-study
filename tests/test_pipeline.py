import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from screening.data import ROOT, SegmentationDataset, validate_rows
from screening.metrics import binary_metrics, structure_loss
from screening.run import evaluate
from screening.run import train

class PipelineTests(unittest.TestCase):
    def test_metric_extremes_and_partial_overlap(self):
        self.assertEqual(binary_metrics([0, 0], [0, 0])['dice'], 1)
        self.assertAlmostEqual(binary_metrics([1, 0], [0, 1])['dice'], 0, places=5)
        self.assertAlmostEqual(binary_metrics([1, 1, 0], [1, 0, 1])['dice'], .5, places=5)

    def test_loss_gradients(self):
        z = torch.zeros(2, 1, 32, 32, requires_grad=True)
        y = torch.zeros_like(z); y[:, :, 8:24, 8:24] = 1
        loss = structure_loss(z, y); loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertLess(z.grad[0, 0, 16, 16].item(), 0)
        self.assertGreater(z.grad[0, 0, 0, 0].item(), 0)

    def test_cwfid_foreground_and_leakage(self):
        ds = SegmentationDataset(ROOT/'data/manifests/cwfid.json', 'test', 64)
        self.assertAlmostEqual(float(ds.originals[0].mean()), 221109/(966*1296), places=6)
        data = json.loads((ROOT/'data/manifests/cwfid.json').read_text())
        self.assertEqual([r['split'] for r in data['samples'] if r['id']=='028'], ['test'])
        with self.assertRaisesRegex(ValueError, 'Duplicate sample'):
            validate_rows([data['samples'][0], data['samples'][0]])

    def test_native_shape_preserved(self):
        class ToyDataset:
            rows = [{'id': 'nonsquare'}]
            originals = [np.ones((12, 20), np.float32)]
            def __len__(self): return 1
            def __getitem__(self, i): return torch.ones(3,32,32), torch.ones(1,32,32), i
        class ToyModel(torch.nn.Module):
            def forward(self, x): return [torch.full((len(x),1,32,32), 20.)]
        with tempfile.TemporaryDirectory() as d:
            result = evaluate(ToyModel(), ToyDataset(), torch.device('cpu'), 1, Path(d))
            import cv2
            self.assertEqual(cv2.imread(str(Path(d)/'predictions/nonsquare.png'),0).shape, (12,20))
            self.assertEqual(result['fixed_threshold']['dice'], 1)
            self.assertLess(result['normalized']['dice'], .001)

    def test_resume_restores_optimizer_and_randomness(self):
        class ToyDataset:
            def __init__(self,*args): pass
            def __len__(self): return 2
            def __getitem__(self,i):
                return torch.rand(3,32,32), torch.ones(1,32,32), i
        def toy_model():
            return torch.nn.Sequential(torch.nn.Conv2d(3,1,1),torch.nn.Dropout2d(.2))
        def val_result(*args,**kwargs):
            return {'normalized': {'dice': .5,'iou': .3}, 'fixed_threshold': {'dice': .5,'iou': .3}}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); manifest=root/'manifest.json'; manifest.write_text('{}')
            cfg=dict(manifest=str(manifest),output=str(root/'continuous'),seed=42,epochs=2,
                     size=32,batch_size=2,lr=.001,weight_decay=.0001,clip=.5,scales=[1],
                     augment=False,scheduler='cosine',amp=False)
            config=root/'config.json';config.write_text(json.dumps(cfg))
            with patch('screening.run.build_model',toy_model), patch('screening.run.SegmentationDataset',ToyDataset), patch('torch.cuda.is_available',return_value=False):
                with patch('screening.run.evaluate',side_effect=val_result): train(config)
                cfg['output']=str(root/'resumed');config.write_text(json.dumps(cfg))
                with patch('screening.run.evaluate',side_effect=[val_result(),RuntimeError('simulated interruption')]):
                    with self.assertRaisesRegex(RuntimeError,'simulated interruption'): train(config)
                with patch('screening.run.evaluate',side_effect=val_result): train(config,resume=True)
            a=torch.load(root/'continuous/last.pt',weights_only=False)
            b=torch.load(root/'resumed/last.pt',weights_only=False)
            for k in a['model']: self.assertTrue(torch.equal(a['model'][k],b['model'][k]),k)
            self.assertEqual(a['optimizer']['param_groups'],b['optimizer']['param_groups'])

if __name__ == '__main__': unittest.main()
