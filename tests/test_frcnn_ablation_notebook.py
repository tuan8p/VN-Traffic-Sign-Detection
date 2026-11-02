"""Execute notebook orchestration with synthetic evidence; no GPU/network/training."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import uuid
import math
import os
import stat
import sys
import subprocess

import pytest

from vn_tsd.config.resolve import resolve_config
from vn_tsd.pipelines.faster_rcnn import sweep
from vn_tsd.pipelines.faster_rcnn.artifacts import write_json
from vn_tsd.pipelines.faster_rcnn.config import validate_config
from vn_tsd.pipelines.faster_rcnn.dataset import content_hash, file_hash
from vn_tsd.utils.io import load_yaml, save_yaml

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/'notebooks/frcnn/frcnn_ablation.ipynb'
PARAMS = {'train.lr': .009606472124890571, 'train.weight_decay': 1.4756984677083597e-05,
          'model.head_dropout': .4, 'online_preprocess.jitter': .1}
CHOICES = {'lr': ('train','lr',.002), 'weight_decay': ('train','weight_decay',.0005),
           'dropout': ('model','head_dropout',.2),
           'jitter': ('online_preprocess','jitter',0.), 'standard': ('model','head','standard')}


def cell(index):
    return ''.join(json.loads(NOTEBOOK.read_text(encoding='utf8'))['cells'][index]['source'])


def functions(index, namespace):
    tree = ast.parse(cell(index))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'notebook-functions','exec'),namespace)


def config():
    return validate_config(resolve_config(ROOT/'configs/pipelines/faster_rcnn_baseline.yaml',ROOT/'configs/shared.yaml',
        ROOT/'configs/runtime/kaggle.yaml',overrides={'data':{'root':'fixture-unused'},'project':{'wandb_project':'sandbox'},'seed':42,
            'train':{'epochs':100,'warmup_epochs':3,'early_stopping_patience':10}}))


def namespace(tmp_path):
    ns=dict(Path=Path,json=json,copy=copy,hashlib=hashlib,uuid=uuid,stat=stat,math=math,sys=sys,
            os=os,subprocess=subprocess,ZipFile=zipfile.ZipFile,ZIP_STORED=zipfile.ZIP_STORED,
            content_hash=content_hash,file_hash=file_hash,validate_config=validate_config,
            load_yaml=load_yaml,save_yaml=save_yaml,write_json=write_json,CHOICES=CHOICES,
            EXPECTED_WINNER=PARAMS,WINNER_ID='ahgf5kge',
            REFERENCE_CAMPAIGN='frcnn_tune_s42_es10_bayes8_sandbox_v1',
            OUTPUT_ROOT=tmp_path,display=lambda *x:None,FileLink=str)
    functions(5,ns);functions(7,ns)
    tree=ast.parse(cell(7))
    worker=next(n for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=='WORKER_CODE')
    ns['WORKER_CODE']=ast.literal_eval(worker.value)
    return ns


@pytest.mark.parametrize('choice',CHOICES)
def test_only_one_training_parameter_changes_and_winner_is_not_mutated(tmp_path,choice):
    ns=namespace(tmp_path);base=config();winner=sweep.apply_parameters(base,PARAMS)
    winner['train']['run_name']='winner';original=copy.deepcopy(winner)
    actual=ns['make_ablation'](winner,base,choice,tmp_path/'cache')
    expected=copy.deepcopy(original);s,f,v=CHOICES[choice];expected[s][f]=v
    expected['train'].pop('run_name');expected['data']['root']=str((tmp_path/'cache').resolve())
    assert actual==expected and winner==original
    assert actual['model']['pretrained'] and actual['eval']['auto_eval_test']
    assert actual['seed']==42 and actual['train']['epochs']==100


def reference_fixture(tmp_path,ns):
    import yaml
    from frcnn_fixtures import make_cache
    from vn_tsd.pipelines.faster_rcnn.dataset import load_cache
    base=config();base['data']['root']=str(make_cache(tmp_path/'cache'))
    winner=sweep.apply_parameters(base,PARAMS)
    _,data=load_cache(winner)
    spec={'method':'bayes','run_cap':8}
    frozen=dict(base_config_sha256=content_hash(base),sweep_sha256=content_hash(spec),
                dataset_sha256=content_hash(data),source_files=sweep.source_identity(),
                runtime=sweep.runtime_identity(),nproc=2,test_policy='shared')
    metrics=dict(status='completed',best_epoch=49,best_val={'mAP50-95':.4905282939017382},config_sha256=content_hash(winner))
    trial=dict(parameters=PARAMS,config_sha256=content_hash(winner),dataset_sha256=content_hash(data))
    bundle=tmp_path/'reference.zip';r=ns['REFERENCE_CAMPAIGN']+'/'
    with zipfile.ZipFile(bundle,'w') as z:
        for name,obj in {'campaign.json':frozen,'sweep_id.json':{'sweep_id':'upxyjxyl'},
                         'trials/ahgf5kge/metrics.json':metrics,'trials/ahgf5kge/sweep_trial.json':trial,
                         'trials/ahgf5kge/dataset_identity.json':data}.items():z.writestr(r+name,json.dumps(obj))
        for name,obj in {'base_config.yaml':base,'sweep.yaml':spec,'trials/ahgf5kge/resolved_config.yaml':winner}.items():
            z.writestr(r+name,yaml.safe_dump(obj))
        z.writestr(r+'trials/ahgf5kge/checkpoints/checkpoint_best.pt',b'not a valid checkpoint: must not load')
    ns['REFERENCE_SHA256']=file_hash(bundle)
    bundle.with_suffix('.zip.sha256').write_text(file_hash(bundle)+'  reference.zip')
    return bundle,base,winner,frozen


def test_reference_hash_is_pinned_and_no_weights_are_loaded(tmp_path):
    ns=namespace(tmp_path);bundle,base,winner,frozen=reference_fixture(tmp_path,ns)
    assert ns['read_reference'](bundle)[:3]==(winner,base,frozen)
    # Updating an upload sidecar cannot bypass the locally pinned reference hash.
    with zipfile.ZipFile(bundle,'a') as z:z.writestr('extra','tampered')
    bundle.with_suffix('.zip.sha256').write_text(file_hash(bundle))
    with pytest.raises(AssertionError,match='bundle/checksum'):ns['read_reference'](bundle)


def test_prepare_cell_with_synthetic_cache_freezes_reference_and_refuses_runtime_drift(tmp_path,monkeypatch):
    ns=namespace(tmp_path);bundle,base,winner,frozen=reference_fixture(tmp_path,ns)
    source=cell(5)
    source=source.replace('Path("/kaggle/input")','INPUT_FIXTURE').replace('Path("/kaggle/working")','OUTPUT_FIXTURE')
    source=source.replace('Path(load_yaml("configs/runtime/kaggle.yaml")["data_root_default"])','CACHE_FIXTURE')
    # Only substitute the reference hash and mount paths; execute the real prepare cell.
    tree=ast.parse(source);assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=='REFERENCE_SHA256')
    pinned=ast.literal_eval(assignment.value);source=source.replace(pinned,file_hash(bundle))
    ns.update(ABLATION='dropout',REFERENCE_BUNDLE=str(bundle),RESTORE_OUTPUT='',
              INPUT_FIXTURE=tmp_path,OUTPUT_FIXTURE=tmp_path/'output',CACHE_FIXTURE=Path(base['data']['root']))
    exec(compile(source,'prepare-ablation','exec'),ns)
    assert ns['cfg']['model']['head_dropout']==.2
    assert ns['WORK'].name=='frcnn_ablation_dropout_s42_v1'
    assert json.loads((ns['WORK']/'request.json').read_text())==ns['request']
    from vn_tsd.pipelines.faster_rcnn import sweep as module
    monkeypatch.setattr(module,'runtime_identity',lambda:{'different':'runtime'})
    with pytest.raises(AssertionError,match='Runtime'):exec(compile(source,'drift','exec'),ns)


@pytest.mark.parametrize('choice',CHOICES)
def test_run_has_independent_id_config_and_export_and_skips_completed(tmp_path,choice):
    ns=namespace(tmp_path);base=config();winner=sweep.apply_parameters(base,PARAMS)
    cfg=ns['make_ablation'](winner,base,choice,tmp_path/'cache')
    work=tmp_path/f'frcnn_ablation_{choice}_s42_v1';work.mkdir()
    request=dict(dataset_sha256=content_hash({'fixture':1}),config=cfg)
    write_json(request,work/'request.json')
    ns.update(cfg=cfg,request=request,NAME=work.name,WORK=work,REPO=ROOT,ABLATION=choice,START_TRAIN=True)
    calls=[]
    def fake_run(command,**kwargs):
        calls.append((command,kwargs))
        assert '--nproc_per_node=2' in command and '--max_restarts=0' in command
        env=kwargs['env'];assert 'WANDB_SWEEP_ID' not in env and env['WANDB_PROJECT']=='sandbox'
        assert env['WANDB_RUN_GROUP']=='frcnn_ablation_s42_v1'
        saved=load_yaml(work/'train_config.yaml');assert saved['train']['run_name'].startswith(work.name+'_')
        directory=work/'run';directory.mkdir();save_yaml(saved,directory/'resolved_config.yaml')
        write_json({'fixture':1},directory/'dataset_identity.json')
        ck=directory/'checkpoints/checkpoint_best.pt';ck.parent.mkdir();ck.write_bytes(b'new fixture checkpoint')
        write_json(dict(status='completed',config_sha256=content_hash(saved),checkpoint_sha256=file_hash(ck),
                        best_val={'mAP50-95':.4}),directory/'metrics.json')
        write_json({'id':env['WANDB_RUN_ID']},directory/'wandb_run.json')
    ns['subprocess']=SimpleNamespace(run=fake_run)
    ns['launch_ablation']();assert len(calls)==1
    ns['launch_ablation']();assert len(calls)==1
    bundles=list(tmp_path.glob('*.zip'));assert len(bundles)==2
    for bundle in bundles:
        assert file_hash(bundle)==bundle.with_suffix('.zip.sha256').read_text().split()[0]
        with zipfile.ZipFile(bundle) as z:
            assert z.testzip() is None
            assert len(z.namelist())==len(set(z.namelist()))
    # Restore uses its own output only, preserving completed evidence and no relaunch.
    backup=work.with_name(work.name+'_original');work.rename(backup)
    ns['restore_output'](bundles[0],work);ns['launch_ablation']();assert len(calls)==1
    (work/'run/checkpoints/checkpoint_best.pt').write_bytes(b'corrupted')
    with pytest.raises(AssertionError):ns['launch_ablation']()
    assert len(calls)==1


def test_failure_exports_and_does_not_silently_retry(tmp_path):
    ns=namespace(tmp_path);work=tmp_path/'frcnn_ablation_lr_s42_v1';work.mkdir()
    cfg=config();request={'config':cfg};write_json(request,work/'request.json')
    ns.update(cfg=cfg,request=request,NAME=work.name,WORK=work,REPO=ROOT,ABLATION='lr',START_TRAIN=True)
    def fail(*args,**kwargs):raise subprocess.CalledProcessError(1,args[0])
    ns['subprocess']=SimpleNamespace(run=fail)
    with pytest.raises(subprocess.CalledProcessError):ns['launch_ablation']()
    assert json.loads((work/'launch_status.json').read_text())['status']=='failed'
    assert list(tmp_path.glob('*.zip'))
    with pytest.raises(AssertionError,match='lỗi/dở'):ns['launch_ablation']()


def test_notebook_worker_and_defaults_are_ready_without_training_resume():
    notebook=json.loads(NOTEBOOK.read_text(encoding='utf8'))
    for c in notebook['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    assert 'a99d2946b55305cea8c6049bd6f581ab709f876a' in cell(2)
    assert 'START_TRAIN = True' in cell(1)
    assert 'sweep", "agent' not in cell(7)
    ns=namespace(Path('unused'));ast.parse(ns['WORKER_CODE'])
    assert 'FasterRCNNPipeline(cfg, work / \'run\').fit()' in ns['WORKER_CODE']
    assert 'torch.load' not in ns['WORKER_CODE'] and 'load_checkpoint' not in ns['WORKER_CODE']


@pytest.mark.parametrize('choice',["jitter","standard"])
def test_extended_choices_preserve_recipe_and_have_effective_model_semantics(tmp_path,choice):
    from vn_tsd.pipelines.faster_rcnn.config import predictor_config
    from vn_tsd.pipelines.faster_rcnn.transforms import describe_effective
    ns=namespace(tmp_path);base=config();winner=sweep.apply_parameters(base,PARAMS)
    cfg=ns['make_ablation'](winner,base,choice,tmp_path/'cache')
    flags=describe_effective(cfg)['effective']
    if choice=='jitter':
        assert flags['enabled'] is True and flags['jitter']==0
        assert predictor_config(cfg)['name']=='custom_mlp'
        assert cfg['model']['head_dropout']==.4
    else:
        assert flags['jitter']==.1 and flags['enabled'] is True
        assert cfg['model']['head_dropout']==.4  # Retained raw field is inactive for standard.
        assert predictor_config(cfg)=={'name':'standard'}
        # Exercise real offline model construction: standard must install no custom predictor/dropout.
        from vn_tsd.pipelines.faster_rcnn.model import build_faster_rcnn,architecture
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        import torch
        model=build_faster_rcnn(cfg,initialize=False)
        assert isinstance(model.roi_heads.box_predictor,FastRCNNPredictor)
        assert model.roi_heads.box_predictor.cls_score.out_features==53
        assert not any(isinstance(m,torch.nn.Dropout) for m in model.roi_heads.box_predictor.modules())
        assert architecture(cfg)['predictor']=={'name':'standard'}


def test_notebook_selectors_and_expected_choices_match():
    ns=namespace(Path('unused'));tree=ast.parse(cell(5))
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=='CHOICES')
    assert ast.literal_eval(node.value)==CHOICES
    for choice in CHOICES: assert '"'+choice+'"' in cell(1)


def test_export_skips_wandb_and_external_symlinks_but_keeps_canonical_files(tmp_path,monkeypatch):
    ns=namespace(tmp_path);work=tmp_path/'ablation';work.mkdir()
    canonical=work/'wandb/run-real/logs/debug.log';canonical.parent.mkdir(parents=True)
    canonical.write_text('canonical log')
    link=work/'wandb/debug.log'
    external=work/'external-link'
    symlink_method=Path.is_symlink
    try:
        link.symlink_to(canonical)
        external.symlink_to(tmp_path/'outside-secret')
    except OSError:
        # Windows may lack symlink privilege; simulate the same Path boundary.
        link.write_text('alias log');external.write_text('must not be exported')
        monkeypatch.setattr(Path,'is_symlink',lambda p: p in (link,external) or symlink_method(p))
    bundle=ns['export_output'](work)
    with zipfile.ZipFile(bundle) as z:
        assert z.read('ablation/wandb/run-real/logs/debug.log')==b'canonical log'
        assert 'ablation/wandb/debug.log' not in z.namelist()
        assert 'ablation/external-link' not in z.namelist()
        receipt=json.loads(z.read('ablation/export_receipt.json'))
        assert set(receipt['skipped_symlinks'])=={'wandb/debug.log','external-link'}
    assert bundle.with_suffix('.zip.sha256').read_text().split()[0]==file_hash(bundle)
