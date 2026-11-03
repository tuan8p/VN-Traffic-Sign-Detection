"""Final-seed notebook execution boundaries; no network/GPU/pretrained training."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import zipfile

import pytest

from test_frcnn_ablation_notebook import namespace as ablation_namespace, reference_fixture, config, PARAMS
from vn_tsd.pipelines.faster_rcnn import sweep
from vn_tsd.pipelines.faster_rcnn.config import validate_config
from vn_tsd.pipelines.faster_rcnn.dataset import content_hash, file_hash
from vn_tsd.config.resolve import resolve_config
from vn_tsd.utils.io import load_yaml, save_yaml
from vn_tsd.pipelines.faster_rcnn.artifacts import write_json

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/'notebooks/frcnn/03_frcnn_final_seeds.ipynb'


def cell(i):
    return ''.join(json.loads(NOTEBOOK.read_text(encoding='utf8'))['cells'][i]['source'])


def namespace(tmp_path):
    ns=ablation_namespace(tmp_path)
    for i in (1,5,7):
        tree=ast.parse(cell(i));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'final-functions','exec'),ns)
    tree=ast.parse(cell(7));node=next(n for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=='WORKER_CODE')
    ns['WORKER_CODE']=ast.literal_eval(node.value)
    return ns


@pytest.mark.parametrize('seed',[42,7,21,123,2026])
def test_final_changes_only_seed_tracking_and_root(tmp_path,seed):
    ns=namespace(tmp_path);winner=sweep.apply_parameters(config(),PARAMS)
    winner['train']['run_name']='old';original=copy.deepcopy(winner)
    actual=ns['make_final'](winner,seed,tmp_path/'cache')
    expected=copy.deepcopy(winner);expected['train'].pop('run_name')
    expected['seed']=seed;expected['data']['root']=str((tmp_path/'cache').resolve())
    expected['project']['wandb_project']='BTL'
    assert actual==expected and winner==original


@pytest.mark.parametrize('bad',[24,0,True,'7',999])
def test_cannot_add_unplanned_seed(tmp_path,bad):
    ns=namespace(tmp_path)
    with pytest.raises(AssertionError):ns['make_final'](sweep.apply_parameters(config(),PARAMS),bad,tmp_path)


def test_frozen_yaml_and_lock_match_winner_recipe_and_keep_shared_batch(tmp_path):
    ns=namespace(tmp_path);winner=sweep.apply_parameters(config(),PARAMS)
    final=validate_config(resolve_config(ROOT/'configs/pipelines/faster_rcnn.yaml',ROOT/'configs/shared.yaml',
        ROOT/'configs/runtime/kaggle.yaml',overrides={'data':{'root':'fixture-unused'},'project':{'wandb_project':'sandbox'},'seed':42}))
    assert final==winner
    lock=json.loads((ROOT/'configs/experiments/faster_rcnn_final_protocol.json').read_text())
    assert not {'model','train','online_preprocess'} & lock.keys()  # YAML is the config authority.
    for key in ('pipeline','baseline','shared','runtime'):
        assert content_hash(load_yaml(ROOT/lock['configuration'][key+'_yaml'])) == lock['configuration'][key+'_sha256']
    assert lock['new_seeds']==[7,21,123,2026] and lock['seed42_run_id']=='ahgf5kge'
    assert lock['source_pin']=='a99d2946b55305cea8c6049bd6f581ab709f876a'
    raw=load_yaml(ROOT/'configs/pipelines/faster_rcnn.yaml')
    assert 'batch_size' not in raw['train'] and 'early_stopping_patience' not in raw['train']
    baseline=load_yaml(ROOT/'configs/pipelines/faster_rcnn_baseline.yaml')
    assert baseline['train']['lr']==.002 and baseline['train']['weight_decay']==.0005
    assert baseline['model']['head_dropout']==.2
    assert not (ROOT/'configs/pipelines/faster_rcnn_final.yaml').exists()
    assert not (ROOT/'vn_tsd/pipelines/faster_rcnn/final_protocol.json').exists()


def test_actual_prepare_cell_synthetic_cache_and_restore_wrong_seed_rejected(tmp_path):
    ns=namespace(tmp_path);bundle,base,winner,frozen=reference_fixture(tmp_path,ns)
    source=cell(5).replace('Path("/kaggle/input")','INPUT_FIXTURE').replace('Path("/kaggle/working")','OUTPUT_FIXTURE')
    source=source.replace('Path(load_yaml("configs/runtime/kaggle.yaml")["data_root_default"])','CACHE_FIXTURE')
    tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=='REFERENCE_SHA256')
    source=source.replace(ast.literal_eval(node.value),file_hash(bundle))
    ns.update(SEEDS=[7,21],REFERENCE_BUNDLE=str(bundle),RESTORE_OUTPUT='',INPUT_FIXTURE=tmp_path,
              OUTPUT_FIXTURE=tmp_path/'output',CACHE_FIXTURE=Path(base['data']['root']))
    exec(compile(source,'final-prepare','exec'),ns)
    jobs=ns['JOBS'];assert [job['seed'] for job in jobs]==[7,21]
    assert jobs[0]['cfg']['seed']==7 and jobs[0]['cfg']['project']['wandb_project']=='BTL'
    assert jobs[0]['request']['protocol_id']=='frcnn_final_v1'
    assert jobs[0]['cfg']['model']['head_dropout']==.4
    first=jobs[0]['WORK'];ns['export_output'](first)
    archived=next((tmp_path/'output').glob('*.zip'))
    # A different seed has a distinct request and cannot reuse this seed's evidence.
    request=dict(jobs[0]['request'],seed=21)
    target=tmp_path/'restore'/first.name
    with pytest.raises(AssertionError,match='seed/config/source'):ns['restore_output'](archived,target,request)
    assert not target.exists()
    repeat=ns['prepare_final'](42)
    assert repeat['request']['repeat_of']=='ahgf5kge' and repeat['cfg']['seed']==42


def test_batched_launches_have_distinct_run_ids_and_export_skip_completed(tmp_path):
    ids=[];names=[];calls=[];jobs=[]
    ns=namespace(tmp_path)
    for seed in (7,21,123,2026,42):
        winner=sweep.apply_parameters(config(),PARAMS)
        cfg=ns['make_final'](winner,seed,tmp_path/'cache')
        work=tmp_path/f'frcnn_final_custom_s{seed}_ep100_bs24_online_v1';work.mkdir()
        request=dict(config=cfg,dataset_sha256=content_hash({'fixture':1}),seed=seed)
        write_json(request,work/'request.json')
        jobs.append(dict(cfg=cfg,request=request,WORK=work,NAME=work.name,seed=seed))
    ns.update(REPO=ROOT,START_TRAIN=True)
    def fake_run(command,**kwargs):
        work=Path(command[-1]);saved=load_yaml(work/'train_config.yaml');seed=saved['seed']
        env=kwargs['env'];calls.append(seed);ids.append(env['WANDB_RUN_ID'])
        assert env['WANDB_PROJECT']=='BTL' and env['WANDB_RUN_GROUP']=='frcnn_final_v1'
        assert env['WANDB_TAGS']==f'faster-rcnn,final,seed{seed}'+(',repeat' if seed==42 else '')
        assert 'WANDB_SWEEP_ID' not in env and '--nproc_per_node=2' in command
        names.append(saved['train']['run_name'])
        directory=work/'run';directory.mkdir();save_yaml(saved,directory/'resolved_config.yaml')
        write_json({'fixture':1},directory/'dataset_identity.json')
        checkpoint=directory/'checkpoints/checkpoint_best.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'fixture weights')
        write_json(dict(status='completed',config_sha256=content_hash(saved),checkpoint_sha256=file_hash(checkpoint),best_val={'mAP50-95':.4}),directory/'metrics.json')
        write_json({'id':env['WANDB_RUN_ID']},directory/'wandb_run.json')
    ns['subprocess']=SimpleNamespace(run=fake_run)
    ns['run_selected_seeds'](jobs[:2]);ns['run_selected_seeds'](jobs[2:4]);ns['run_selected_seeds'](jobs[4:])
    ns['run_selected_seeds'](jobs)  # Verified completed outputs skip all five.
    for job in jobs:
        work=job['WORK']
        bundle=next(tmp_path.glob(work.name+'*.zip'))
        assert file_hash(bundle)==bundle.with_suffix('.zip.sha256').read_text().split()[0]
        with zipfile.ZipFile(bundle) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    assert len(calls)==len(set(ids))==len(set(names))==5
    assert calls==[7,21,123,2026,42]


@pytest.mark.parametrize('value,expected',[(21,[21]),([21,7],[21,7]),([42],[42]),
    ([7,21,123,2026],[7,21,123,2026]),([42,7,21,123,2026],[42,7,21,123,2026])])
def test_seed_selection_preserves_order(tmp_path,value,expected):
    assert namespace(tmp_path)['normalize_seeds'](value)==expected


@pytest.mark.parametrize('value',[set([21,7]),[],[7,7],[24],[True],[7,'21'],[7]*6,42.0,'42'])
def test_invalid_queue_rejected_before_bootstrap(tmp_path,value):
    with pytest.raises(AssertionError):namespace(tmp_path)['normalize_seeds'](value)


def test_queue_stops_on_failure_and_does_not_launch_later_seed(tmp_path):
    ns=namespace(tmp_path);calls=[]
    def launch(**job):
        calls.append(job['seed'])
        if job['seed']==21:raise RuntimeError('controlled failure')
    ns['launch_final']=launch
    with pytest.raises(RuntimeError):ns['run_selected_seeds']([{'seed':7},{'seed':21},{'seed':123}])
    assert calls==[7,21]


def test_restore_mapping_rejects_ambiguous_single_bundle_for_batch(tmp_path):
    normalize=namespace(tmp_path)['normalize_restore']
    assert normalize('seed7.zip',[7])=={7:'seed7.zip'}
    assert normalize({21:'seed21.zip'},[7,21])=={21:'seed21.zip'}
    for value in ('seed7.zip',{42:'seed42.zip'},{True:'seed.zip'},{7:None}):
        with pytest.raises(AssertionError):normalize(value,[7,21])


def test_final_failure_exports_without_retry(tmp_path):
    ns=namespace(tmp_path);work=tmp_path/'final';work.mkdir();cfg=config();write_json({'fixture':1},work/'request.json')
    ns.update(cfg=cfg,WORK=work,NAME='frcnn_final_test',REPO=ROOT,START_TRAIN=True,SEED=7)
    def fail(*args,**kwargs):raise subprocess.CalledProcessError(1,args[0])
    ns['subprocess']=SimpleNamespace(run=fail)
    job=dict(cfg=cfg,request={},WORK=work,NAME='frcnn_final_test',seed=7)
    with pytest.raises(subprocess.CalledProcessError):ns['launch_final'](**job)
    assert json.loads((work/'launch_status.json').read_text())['status']=='failed'
    assert list(tmp_path.glob('*.zip.sha256'))
    with pytest.raises(AssertionError,match='lỗi/dở'):ns['launch_final'](**job)


def test_notebook_defaults_and_worker_do_not_resume():
    n=json.loads(NOTEBOOK.read_text(encoding='utf8'))
    for c in n['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['outputs']==[]
    assert 'SEED = [7, 21]' in cell(1) and 'START_TRAIN = True' in cell(1)
    assert 'a99d2946b55305cea8c6049bd6f581ab709f876a' in cell(2)
    ns=namespace(Path('unused'));ast.parse(ns['WORKER_CODE'])
    assert 'load_checkpoint' not in ns['WORKER_CODE'] and 'torch.load' not in ns['WORKER_CODE']
    assert 'ABLATION' not in ''.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
