"""Run parameter-matched, multi-seed experiments and report sample standard deviation."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from nlp_lm.models import MODEL_NAMES, build_model

SETTINGS = {
    'fnn': (64, 99),
    'rnn': (64, 136),
    'lstm': (64, 123),
    'transformer': (96, 192),
}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(root, manifest):
    rows = []
    histories = {name: [] for name in MODEL_NAMES}
    for seed in manifest['seeds']:
        for name in MODEL_NAMES:
            folder = root / f'seed_{seed}' / name
            if not (folder / 'summary.json').exists():
                continue
            summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
            if summary['completed_epochs'] != manifest['epochs'] or summary['debug_run']:
                continue
            rows.append(dict(seed=seed, **summary))
            with (folder / 'history.csv').open(encoding='utf-8') as f:
                histories[name].append(list(csv.DictReader(f)))
    if not rows:
        return
    write_csv(root / 'per_seed.csv', rows)
    aggregate = []
    for name in MODEL_NAMES:
        group = [r for r in rows if r['model'] == name]
        if not group:
            continue
        values = [r['val_ppl'] for r in group]
        aggregate.append(dict(model=name, parameters=group[0]['parameters'],
            completed_seeds=len(group), mean_val_ppl=statistics.mean(values),
            sample_std_val_ppl=statistics.stdev(values) if len(values)>1 else None,
            min_val_ppl=min(values), max_val_ppl=max(values),
            mean_best_epoch=statistics.mean(r['best_epoch'] for r in group),
            mean_train_seconds=statistics.mean(r['total_train_seconds'] for r in group)))
    write_csv(root / 'aggregate.csv', aggregate)
    complete = len(rows) == len(manifest['seeds'])*len(MODEL_NAMES)
    lines = ['# Parameter-matched multi-seed experiment', '',
        f"Status: {'COMPLETE' if complete else 'INCOMPLETE'} ({len(rows)}/{len(manifest['seeds'])*4} runs)", '',
        f"Seeds: {manifest['seeds']}; epochs: {manifest['epochs']}; batch size: 128; sequence length: 32.",
        'Adam learning rate: 0.001; dropout: 0.1; layers: 1; attention heads: 4.',
        'Full training split; fixed final 1,000 sentences for validation; character-level PPL.',
        'Each run selects its checkpoint by lowest validation loss. The reported mean and sample SD (ddof=1) are over the best validation PPL of each seed, not over epochs.',
        'Validation is used for checkpoint selection; these are not independent test-set results.',
        'Parameter matching does not match compute or optimize hyperparameters for each architecture.',
        'The original small_10_gpu experiment used batch size 32; this experiment uses 128. Compare architectures within this experiment.', '',
        '| Model | Embedding | Hidden / FFN | Parameters | Seeds | Best validation PPL (mean ± sample SD) | Mean train seconds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in aggregate:
        e,h=SETTINGS[r['model']]
        sd=r['sample_std_val_ppl']
        spread=f'{sd:.3f}' if sd is not None else 'N/A'
        lines.append(f"| {r['model']} | {e} | {h} | {r['parameters']:,} | {r['completed_seeds']} | {r['mean_val_ppl']:.3f} ± {spread} | {r['mean_train_seconds']:.1f} |")
    lines += ['', '## Per-seed results', '', '| Model | Seed | Best epoch | Validation PPL |', '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['seed']} | {r['best_epoch']} | {r['val_ppl']:.4f} |")
    (root / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if complete:
        os.environ.setdefault('MPLCONFIGDIR', str(root.resolve()/'matplotlib-cache'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes=plt.subplots(1,2,figsize=(12,4.5))
        axes[0].bar([r['model'].upper() for r in aggregate], [r['mean_val_ppl'] for r in aggregate], yerr=[r['sample_std_val_ppl'] for r in aggregate], capsize=5)
        axes[0].set(xlabel='Model',ylabel='Best validation PPL',title='Mean and sample SD across seeds')
        for name, runs in histories.items():
            means=[statistics.mean(float(run[i]['val_ppl']) for run in runs) for i in range(manifest['epochs'])]
            sds=[statistics.stdev(float(run[i]['val_ppl']) for run in runs) for i in range(manifest['epochs'])]
            xs=list(range(1,manifest['epochs']+1))
            axes[1].plot(xs,means,label=name.upper())
            axes[1].fill_between(xs,[m-s for m,s in zip(means,sds)],[m+s for m,s in zip(means,sds)],alpha=.15)
        axes[1].set(xlabel='Epoch',ylabel='Validation PPL',title='Validation curves: mean ± sample SD')
        axes[1].legend()
        for ax in axes: ax.grid(axis='y',alpha=.3)
        fig.tight_layout()
        fig.savefig(root/'comparison.png',dpi=180)
        plt.close(fig)
    print('\n'.join(lines[:19]),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/matched_1m_3seeds'))
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--summarize-only',action='store_true')
    args=p.parse_args()
    if args.epochs<1 or len(set(args.seeds))!=len(args.seeds) or len(args.seeds)<2:
        p.error('Use positive epochs and at least two unique seeds')
    import torch
    bundle=torch.load('data/processed.pt',map_location='cpu',weights_only=True)
    manifest=dict(seeds=args.seeds,epochs=args.epochs,batch_size=128,seq_len=32,lr=.001,dropout=.1,layers=1,heads=4,data_stats=bundle['stats'],models={})
    for name,(e,h) in SETTINGS.items():
        cfg=dict(embedding_dim=e,hidden_dim=h,seq_len=32,layers=1,heads=4,dropout=.1)
        count=sum(p.numel() for p in build_model(name,len(bundle['vocab']),cfg).parameters() if p.requires_grad)
        if abs(count-1_000_000)>50_000:
            raise ValueError(f'{name}: {count} outside 1M ±5% budget')
        manifest['models'][name]=dict(**cfg,parameters=count)
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Existing manifest differs; use a new output folder')
    path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    if not args.summarize_only:
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        for seed in args.seeds:
            for name,(e,h) in SETTINGS.items():
                folder=args.output/f'seed_{seed}'/name
                if folder.exists() and any(folder.iterdir()):
                    saved=json.loads((folder/'config.json').read_text(encoding='utf-8'))
                    expected=dict(seed=seed,model=name,embedding_dim=e,hidden_dim=h,epochs=args.epochs,batch_size=128,seq_len=32,layers=1,heads=4,dropout=.1,lr=.001,device='cuda',data_stats=bundle['stats'])
                    if any(saved[k]!=v for k,v in expected.items()): raise ValueError(f'Configuration mismatch: {folder}')
                    summary_path=folder/'summary.json'
                    if summary_path.exists():
                        summary=json.loads(summary_path.read_text(encoding='utf-8'))
                        if summary['completed_epochs']==args.epochs and not summary['debug_run']:
                            print(f'SKIP completed: {folder}',flush=True)
                            continue
                    raise RuntimeError(f'Incomplete run preserved: {folder}; use a new output folder')
                command=[sys.executable,'-u','train.py','--model',name,'--seed',str(seed),'--epochs',str(args.epochs),'--batch-size','128','--seq-len','32','--embedding-dim',str(e),'--hidden-dim',str(h),'--layers','1','--heads','4','--dropout','0.1','--lr','0.001','--device','cuda','--log-every','0','--output',str(args.output/f'seed_{seed}')]
                print('RUN '+' '.join(command),flush=True)
                subprocess.run(command,check=True)
                collect(args.output,manifest)
    collect(args.output,manifest)


if __name__=='__main__':
    main()
