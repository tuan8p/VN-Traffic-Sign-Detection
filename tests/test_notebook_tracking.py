import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from vn_tsd.eval.metrics import summarize_yolo


def test_notebook_logs_epoch_once_and_keeps_missing_ar100(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'notebooks/04_train_yolo26m.ipynb'
    notebook = json.loads(path.read_text())
    callback = None
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        compile(tree, str(path), 'exec')
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'log_epoch':
                callback = node
    assert callback is not None
    run = Mock()
    env = dict(pending_epochs={1}, summarize_yolo=summarize_yolo,
               history=[], save_json=Mock(), run_dir=tmp_path, run=run)
    exec(compile(ast.Module(body=[callback], type_ignores=[]), '<callback>', 'exec'), env)
    trainer = SimpleNamespace(
        epoch=0, validator=SimpleNamespace(metrics=SimpleNamespace(
            box=SimpleNamespace(map=0.4, map50=0.7, map75=0.45))),
        tloss=None, label_loss_items=lambda *args, **kwargs: {'train/box_loss': 1.2},
        metrics={'val/box_loss': 1.0}, lr={'pg0': 0.01},
    )
    env['log_epoch'](trainer)
    env['log_epoch'](trainer)
    trainer.epoch = 1
    env['log_epoch'](trainer)
    assert run.log.call_count == 1
    assert run.log.call_args.args[0]['val/mAP75'] == 0.45
    assert 'val/AR100' not in run.log.call_args.args[0]
    assert env['history'][0]['val/AR100'] is None
