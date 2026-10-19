"""RCSE-Lite unit checks: routing, masks, losses, benchmark interfaces, checkpoints."""
import tempfile
import unittest
from pathlib import Path
import torch
from bchrpp.full_model import BCHRPP, EvidenceEncoder, CoupledEvidenceRouter, masked_softmax
from bchrpp.legacy_model import BCHRPP as LegacyBCHRPP
from bchrpp.enhanced_v1 import BCHRPP as EnhancedBCHRPP
from losses.bchrpp_loss import compute_loss
from metrics.evaluation import mosi_metrics, simsv2_metrics


class TestRCSELite(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    @staticmethod
    def model(output_dim=1, **kwargs):
        return BCHRPP(d=16, output_dim=output_dim, queries=2, text_dim=12,
                      audio_dim=5, vision_dim=8, dropout=0.,modality_dropout=0.,**kwargs)

    @staticmethod
    def examples():
        xs=[torch.randn(3,6,d) for d in (12,5,8)]
        masks={k:torch.ones(3,6,dtype=torch.bool) for k in ('text','audio','vision')}
        masks['audio'][0,:]=False
        masks['vision'][1,:]=False
        masks['audio'][2,:]=False
        masks['vision'][2,:]=False
        masks['text'][0,3:]=False
        return xs,masks

    def test_masked_softmax_empty_and_masked(self):
        z=masked_softmax(torch.tensor([[1.,2.,3.],[1.,2.,3.]]),
                         torch.tensor([[1,0,1],[0,0,0]],dtype=torch.bool))
        self.assertAlmostEqual(z[0].sum().item(),1.,places=6)
        self.assertEqual(z[0,1].item(),0.)
        self.assertEqual(z[1].sum().item(),0.)

    def test_encoder_empty_and_padding(self):
        encoder=EvidenceEncoder(d=16,q=3,dropout=0).eval()
        x=torch.randn(3,6,16)
        mask=torch.tensor([[1,1,0,0,0,0],[1,1,1,1,0,0],[0,0,0,0,0,0]],dtype=torch.bool)
        expected=encoder(x,mask)
        edited=x.clone();edited[~mask]=1e5
        torch.testing.assert_close(encoder(edited,mask),expected,atol=1e-6,rtol=1e-6)
        self.assertEqual(expected[-1].abs().max().item(),0.)

    def test_router_missing_single_modality(self):
        router=CoupledEvidenceRouter(16)
        reps=torch.randn(3,3,16)
        valid=torch.tensor([[1,0,0],[1,1,0],[1,1,1]],dtype=torch.bool)
        probs,conf,comp=router(reps,valid)
        self.assertTrue(torch.all(probs[~valid] == 0))
        self.assertTrue(torch.allclose(probs.sum(-1),torch.ones(3)))
        self.assertEqual(probs[0,0].item(),1.)
        self.assertEqual(comp[0].abs().sum().item(),0.)

    def test_router_complement_is_reliability_discounted(self):
        router=CoupledEvidenceRouter(4)
        with torch.no_grad():
            router.quality_head.weight.zero_();router.quality_head.bias.fill_(-8.)
        w,confidence,complement=router(torch.randn(2,3,4),torch.ones(2,3,dtype=torch.bool))
        self.assertLess(float(complement.detach().max()),.001)

    def test_regression_backprop_full(self):
        model=self.model()
        xs,masks=self.examples()
        result,aux=model(*xs,masks=masks,return_aux=True)
        self.assertEqual(result.shape,(3,1))
        self.assertEqual(aux['modality_weights'].shape,(3,3))
        self.assertTrue((aux['modality_weights'][~aux['modality_available']]==0).all())
        self.assertTrue(torch.allclose(aux['modality_weights'].sum(-1),torch.ones(3)))
        loss,parts=compute_loss(result,torch.tensor([[.6],[-.3],[.2]]),aux)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name in ('text_projection','audio_projection','vision_projection','router',
                     'reasoner','arbiter','probe'):
            mod=getattr(model,name)
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                                for p in mod.parameters()),name)

    def test_classification_backprop(self):
        model=self.model(output_dim=3)
        xs,masks=self.examples()
        result,aux=model(*xs,masks=masks,return_aux=True)
        loss,_=compute_loss(result,torch.tensor([0,1,2]),aux,task='classification')
        self.assertEqual(result.shape,(3,3))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_masked_values_do_not_change_prediction(self):
        model=self.model().eval()
        xs,masks=self.examples()
        modified=[x.clone() for x in xs]
        for name,m in zip(('text','audio','vision'),modified):
            m[~masks[name]]=1e5
        with torch.no_grad():
            a=model(*xs,masks=masks)
            b=model(*modified,masks=masks)
        torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)

    def test_absent_modality_content_invariant(self):
        model=self.model().eval()
        xs,masks=self.examples()
        x2=[x.clone() for x in xs]
        x2[1][0,:]=1e6
        with torch.no_grad():
            a=model(*xs,masks=masks)
            b=model(*x2,masks=masks)
        torch.testing.assert_close(a,b)

    def test_all_modalities_missing_rejected(self):
        model=self.model()
        xs,masks=self.examples()
        for m in masks.values(): m[0,:]=False
        with self.assertRaisesRegex(ValueError,'at least one modality'):
            model(*xs,masks=masks)

    def test_modality_dropout_never_drops_everything(self):
        model=BCHRPP(
            d=16,queries=2,text_dim=12,audio_dim=5,vision_dim=8,
            modality_dropout=.99,dropout=0.)
        xs,masks=self.examples()
        for _ in range(8):
            _,aux=model(*xs,masks=masks,return_aux=True)
            self.assertTrue(aux['modality_available'].any(-1).all())
            self.assertTrue(torch.isfinite(aux['modality_weights']).all())

    def test_flags_and_dense_reasoning(self):
        xs,masks=self.examples()
        for kwargs in ({'reliability':False,'complementarity':False},
                       {'sparse_graph':False},
                       {'temporal':False,'refinement':False}):
            model=self.model(**kwargs).eval()
            with torch.no_grad(): out=model(*xs,masks=masks)
            self.assertTrue(torch.isfinite(out).all())

    def test_new_checkpoint_round_trip(self):
        xs,masks=self.examples()
        model=self.model().eval()
        with tempfile.TemporaryDirectory() as td:
            ck=Path(td)/'model.pt'
            torch.save({'version':'bchrpp_rcse_lite_v1','model_state':model.state_dict(),
                        'model_config':dict(d=16,queries=2,text_dim=12,audio_dim=5,vision_dim=8,
                                            dropout=0.,modality_dropout=0.)},ck)
            blob=torch.load(ck,weights_only=True,map_location='cpu')
            clone=BCHRPP(**blob['model_config']).eval()
            clone.load_state_dict(blob['model_state'],strict=True)
            with torch.no_grad():
                torch.testing.assert_close(clone(*xs,masks=masks),model(*xs,masks=masks))

    def test_historical_checkpoint_loads_legacy(self):
        file=Path(__file__).resolve().parents[1]/'checkpoints'/'mosi_best.pt'
        if not file.exists():self.skipTest('source historical checkpoint not included')
        ck=torch.load(file,weights_only=True,map_location='cpu')
        reference=LegacyBCHRPP(**ck['model_config'])
        reference.load_state_dict(ck['model_state'],strict=True)
        self.assertIn('mae',ck['valid_metrics'])

    def test_enhanced_v1_model_loadable(self):
        reference=EnhancedBCHRPP(d=16,queries=2,text_dim=12,audio_dim=5,vision_dim=8,
                                  modality_dropout=0.,dropout=0.)
        xs,masks=self.examples()
        with torch.no_grad():
            result=reference.eval()(*xs,masks=masks)
        self.assertTrue(torch.isfinite(result).all())

    def test_metric_functions(self):
        self.assertAlmostEqual(mosi_metrics([-.6,0,.7],[-1,0,1])['acc2'],1.)
        self.assertAlmostEqual(simsv2_metrics([-.3,0,.5],[-.3,0,.5])['acc5_percent'],100.)

if __name__ == '__main__': unittest.main()
