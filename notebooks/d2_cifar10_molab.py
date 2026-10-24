# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "torch", "torchvision", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "tqdm"]
# ///
import marimo

app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    return (mo,)

@app.cell
def _(mo):
    mo.md("""
    # D² Pruning · CIFAR-10 · Molab

    Đặt thư mục **cifar-10-batches-py/** cạnh notebook trong working directory.
    Gắn GPU từ **notebook specs** trước khi chạy. Dữ liệu được kiểm tra bằng
    torchvision với `download=False`; không tải lại CIFAR-10.

    Pipeline: baseline → forgetting scores/embeddings → Random và D² → CSV.
    Mặc định một seed, pruning 70%, 40.000 bước cho mỗi model mới.
    Có thể sửa cấu hình bên dưới trước khi bấm chạy. Log xuất hiện trong cell chạy.

    **Giới hạn:** hard cutoff `mis_ratio=0` là cấu hình thử nghiệm chưa được xác nhận
    từ bài báo. Không coi chênh lệch accuracy nhỏ là bằng chứng reproduce chính xác.
    Giữ nguyên thuật toán công khai tại commit bf1ea46b22cc79549df421fcd92c9cd2d3408008.

    Giữ cell kết nối Google Drive hiện có tạo biến **fs**. Cell chạy phụ thuộc vào fs.
    Notebook kiểm tra quyền ghi trước training, sao lưu vào `.ipynb_checkpoints/d2-<run-id>/`
    sau trích xuất baseline và khi hoàn tất. Bao gồm CSV, checkpoint, scores, features,
    indices và logs. Không tải lại CIFAR-10.
    [Bài báo](https://arxiv.org/html/2310.07931v1) ·
    [Molab storage](https://docs.marimo.io/guides/molab/#storage)
    """)
    return

@app.cell
def _():
    config = {
        "data_path": "cifar-10-batches-py/",
        "output_path": "d2-reproduction/",
        "seeds": [0],
        "pruning_rates": [70],  # mở rộng: [30, 50, 70, 80, 90]
        "steps": 40000,
        "mis_ratio": {30: 0.0, 50: 0.0, 70: 0.0, 80: 0.0, 90: 0.0},
        "existing_baseline": None,
        "drive_path": ".ipynb_checkpoints/",
    }
    # existing_baseline: thư mục chứa ckpt-last.pt và td-all-data.pickle nếu có.
    # Nếu thay cấu hình, dùng output_path mới để tránh dùng nhầm kết quả cũ.
    # Baseline Kaggle cũ có 39.200 bước; việc dùng lại cần ghi rõ trong báo cáo.
    return (config,)

@app.cell
def _(mo):
    start = mo.ui.run_button(label="Chạy reproduction")
    start
    return (start,)

@app.cell
def _():
    def run_reproduction(config, fs):
        from pathlib import Path
        import os, sys, subprocess, json, shutil, hashlib, re
        import numpy as np
        import pandas as pd
        import torch
        from datetime import datetime, timezone
        import posixpath
        if getattr(fs, 'protocol', None) in ['file', 'local', ('file', 'local')]:
            raise ValueError('fs phải là kết nối Google Drive, không phải local filesystem')
        drive_root = config['drive_path'].rstrip('/')
        # Xác nhận kết nối và thư mục đích trước khi bắt đầu training.
        fs.makedirs(drive_root, exist_ok=True)
        fs.ls(drive_root)
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        remote_run = posixpath.join(drive_root, 'd2-' + run_id)
        fs.makedirs(remote_run, exist_ok=True)
        probe = posixpath.join(remote_run, 'storage-check.json')
        with fs.open(probe, 'wb') as stream:
            stream.write(b'{"write_test": true}')
        with fs.open(probe, 'rb') as stream:
            assert stream.read() == b'{"write_test": true}'
        print('Google Drive destination:', remote_run)
        SOURCE = Path(config['data_path']).expanduser().resolve()
        ROOT = Path(config['output_path']).expanduser().resolve()
        REPO = ROOT / 'repo'
        DATA = ROOT / 'data'
        RESULTS = ROOT / 'results'
        COMMIT = 'bf1ea46b22cc79549df421fcd92c9cd2d3408008'
        SEEDS = config['seeds']  # dùng [0, 1, 2] cho đánh giá nhiều seed
        PRUNING_RATES = config['pruning_rates']  # có thể dùng [30, 50, 70, 80, 90]
        STEPS = config['steps']  # cùng ngân sách cho full-data và các tập con
        # Cutoff thử nghiệm, KHÔNG phải giá trị đã xác nhận từ bài báo.
        # Không áp dụng 0.4 của ví dụ README cho mọi pruning rate.
        MIS_RATIO = config['mis_ratio']
        # Có thể gắn output baseline cũ làm input rồi điền thư mục chứa ckpt-last.pt và td-all-data.pickle.
        EXISTING_BASELINE = config['existing_baseline']
        assert torch.cuda.is_available(), 'Gắn GPU từ notebook specs của Molab'
        assert SOURCE.is_dir(), f'Không tìm thấy {SOURCE}'
        for name in [f'data_batch_{i}' for i in range(1,6)] + ['test_batch', 'batches.meta']:
            assert (SOURCE / name).is_file(), name
        for p in [ROOT, DATA / 'cifar10', RESULTS]: p.mkdir(parents=True, exist_ok=True)
        link = DATA / 'cifar10' / 'cifar-10-batches-py'
        if not link.exists(): link.symlink_to(SOURCE, target_is_directory=True)
        print(torch.__version__, torch.cuda.get_device_name(0))
        def backup():
            entries = []
            for local in sorted(RESULTS.rglob('*')):
                if not local.is_file(): continue
                target = posixpath.join(remote_run, local.relative_to(RESULTS).as_posix())
                fs.makedirs(posixpath.dirname(target), exist_ok=True)
                fs.put_file(str(local), target)
                info = fs.info(target)
                if info.get('size') is not None and int(info['size']) != local.stat().st_size:
                    raise IOError(f'Upload size mismatch: {target}')
                entries.append({'path': target, 'bytes': local.stat().st_size})
            with fs.open(posixpath.join(remote_run, 'backup-manifest.json'), 'wb') as stream:
                stream.write(json.dumps(entries, indent=2).encode())
            print('Đã sao lưu lên Google Drive:', remote_run)



        def command(args, cwd=None):
            subprocess.run([str(x) for x in args], cwd=cwd, check=True)
        if not REPO.exists():
            command(['git', 'clone', 'https://github.com/adymaharana/d2pruning.git', REPO])
        command(['git', 'checkout', '--detach', COMMIT], cwd=REPO)
        # Dùng thư viện sẵn có của Kaggle thay vì ép scikit-learn 0.24 trên Python mới.
        import scipy, sklearn, torchvision, matplotlib, tqdm
        from torchvision.datasets import CIFAR10
        train_data = CIFAR10(str(DATA / 'cifar10'), train=True, download=False)
        test_data = CIFAR10(str(DATA / 'cifar10'), train=False, download=False)
        assert len(train_data)==50000 and len(test_data)==10000
        print('Dataset verified:', len(train_data), len(test_data))


        def patch(file, old, new):
            p = REPO / file
            s = p.read_text()
            if old in s: p.write_text(s.replace(old, new))
            elif new not in s: raise RuntimeError(f'Không tìm thấy vị trí patch: {file}: {old}')
        for file in ['train.py', 'generate_importance_score.py']:
            patch(file, 'num_workers=16', 'num_workers=0')
        patch('core/data/MiscDataset.py', 'download=True', 'download=False')
        patch('train.py', 'start_time = datetime.now()', 'start_time = time.time()')
        patch('train.py', 'trainer.train(current_epoch, -1,', 'trainer.train(current_epoch, iterations_epoch,')
        patch('train.py', 'num_of_iterations -= iterations_per_epoch', 'num_of_iterations -= iterations_epoch')
        patch('selection_mp.py', 'coreset_index.numpy().astype(np.int32)', 'np.asarray(coreset_index).astype(np.int32)')
        (REPO.parent / 'data-pruning-analysis').mkdir(exist_ok=True)
        # Seed phải được đặt trong subprocess thực thi train.py.
        runner_source = "import os, sys, random, runpy\nimport numpy as np\nimport torch\nseed = int(os.environ['D2_SEED'])\nrandom.seed(seed)\nnp.random.seed(seed)\ntorch.manual_seed(seed)\ntorch.cuda.manual_seed_all(seed)\ntorch.backends.cudnn.benchmark = False\ntorch.backends.cudnn.deterministic = True\nscript = sys.argv.pop(1)\nrunpy.run_path(script, run_name='__main__')\n"
        compile(runner_source, "seeded_run.py", "exec")
        (REPO / "seeded_run.py").write_text(runner_source)
        def run(script, args, seed):
            env = dict(os.environ, D2_SEED=str(seed), PYTHONHASHSEED=str(seed), OMP_NUM_THREADS='2')
            from collections import deque
            tail = deque(maxlen=80)
            with subprocess.Popen([sys.executable, '-u', 'seeded_run.py', script, *map(str,args)],
                                  cwd=REPO, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    tail.append(line)
                status = process.wait()
            if status:
                raise RuntimeError(f'{script} exited {status}:\n' + ''.join(tail))
        manifest = dict(commit=COMMIT, seeds=SEEDS, pruning_rates=PRUNING_RATES,
                        steps=STEPS, mis_ratio=MIS_RATIO, cutoff_verified=False,
                        torch=torch.__version__, torchvision=torchvision.__version__, sklearn=sklearn.__version__)
        (RESULTS / 'manifest.json').write_text(json.dumps(manifest, indent=2))


        import pickle
        baselines = {}
        for seed in SEEDS:
            base = RESULTS / f'seed-{seed}'
            folder = base / 'all-data'
            folder.mkdir(parents=True, exist_ok=True)
            if EXISTING_BASELINE:
                for name in ['ckpt-last.pt', 'td-all-data.pickle']:
                    src = Path(EXISTING_BASELINE) / name
                    assert src.is_file(), src
                    if not (folder / name).exists(): shutil.copy2(src, folder / name)
            elif not (folder / 'complete.json').exists():
                assert not (folder / 'log-train-all-data.log').exists(), 'Run chưa hoàn tất: dùng ROOT mới để chạy lại'
                run('train.py', ['--dataset','cifar10','--data-dir',DATA,'--gpuid','0',
                    '--iterations',STEPS,'--lr','0.1','--network','resnet18','--batch-size','256',
                    '--task-name','all-data','--base-dir',base], seed)
                (folder / 'complete.json').write_text(json.dumps(manifest))
            score = folder / 'data-score-all-data.pickle'
            feature = folder / 'train-features-conv3-all-data.npy'
            if not (folder / 'features-complete.json').exists():
                run('generate_importance_score.py', ['--dataset','cifar10','--data-dir',DATA,
                    '--gpuid','0','--base-dir',base,'--task-name','all-data','--feature'], seed)
                (folder / 'features-complete.json').write_text('{}')
            with score.open('rb') as f: scores = pickle.load(f)
            features = np.load(feature, mmap_mode='r')
            assert len(scores['forgetting']) == 50000 and features.shape[0] == 50000
            assert np.isfinite(features).all()
            baselines[seed] = (base, score, feature)
            print(seed, features.shape)
            backup()


        PARAMS = {30:(10,0.9), 50:(5,1.0), 70:(1,0.1), 80:(2,0.0), 90:(2,0.0)}
        for seed, (base, score, feature) in baselines.items():
            for pruning in PRUNING_RATES:
                ratio = (100-pruning)/100
                k, gamma = PARAMS[pruning]
                assert 0 <= MIS_RATIO[pruning] < 1-ratio
                for method in ['random','d2']:
                    task = f'{method}-prune-{pruning}'
                    folder = base / task
                    if (folder / 'complete.json').exists(): continue
                    assert not folder.exists(), f'Run chưa hoàn tất: {folder}; dùng ROOT mới'
                    args = ['--dataset','cifar10','--data-dir',DATA,'--gpuid','0',
                            '--iterations',STEPS,'--batch-size','256','--lr','0.1','--network','resnet18',
                            '--base-dir',base,'--task-name',task,'--coreset','--coreset-ratio',ratio,
                            '--data-score-path',score,'--coreset-key','forgetting','--label-balanced']
                    if method == 'random': args += ['--coreset-mode','random']
                    else: args += ['--coreset-mode','class','--budget-mode','uniform','--sampling-mode','graph',
                                  '--feature-path',feature,'--mis-ratio',MIS_RATIO[pruning],
                                  '--n-neighbor',k,'--gamma',gamma,'--stratas','25',
                                  '--graph-mode','sum','--graph-sampling-mode','weighted']
                    run('train.py', args, seed)
                    idx = np.load(folder / f'coreset-{task}.npy').astype(int)
                    assert len(idx) == round(50000*ratio) and len(np.unique(idx)) == len(idx)
                    assert idx.min() >= 0 and idx.max() < 50000
                    print('Class counts:', np.bincount(np.array(train_data.targets)[idx], minlength=10))
                    (folder / 'complete.json').write_text(json.dumps(dict(seed=seed, method=method,
                        pruning=pruning, steps=STEPS, mis_ratio=MIS_RATIO[pruning] if method=='d2' else None)))


        backup()
        paper = {'random':{0:95.5,30:94.3,50:93.4,70:90.9,80:88.0,90:79.0},
                 'd2':{30:95.7,50:94.9,70:93.3,80:91.4,90:87.1}}
        rows=[]
        for seed in SEEDS:
            base=RESULTS / f'seed-{seed}'
            tasks=[('random',0,'all-data')] + [(m,p,f'{m}-prune-{p}') for p in PRUNING_RATES for m in ['random','d2']]
            for method,pruning,task in tasks:
                log=base/task/f'log-train-{task}.log'
                if not log.exists(): continue
                text=log.read_text()
                best=re.findall(r'Best acc: ([0-9.]+)',text)
                last=re.findall(r'Test Accuracy: ([0-9.]+)',text)
                if not best or not last: continue
                best_pct=float(best[-1])*100  # dòng Best acc cuối là tỷ lệ 0..1
                rows.append(dict(seed=seed,method='full' if pruning==0 else method,pruning=pruning,
                                 best_accuracy=best_pct,last_accuracy=float(last[-1]),
                                 paper_accuracy=paper[method][pruning],delta_pp=best_pct-paper[method][pruning]))
        results=pd.DataFrame(rows)
        results.to_csv(RESULTS/'results.csv',index=False)
        print(results.to_string(index=False))
        if len(SEEDS)>1:
            summary=results.groupby(['method','pruning']).agg(mean=('best_accuracy','mean'),std=('best_accuracy','std'),n=('seed','count'))
            print(summary.to_string())
            summary.to_csv(RESULTS/'summary.csv')
        print('Lưu toàn bộ output:', RESULTS)
        print('Kết quả D2 dùng cutoff thử nghiệm; chênh lệch số học không tự chứng minh tương đương thống kê.')

        backup()
        return results, remote_run
    return (run_reproduction,)

@app.cell
def _(config, fs, mo, run_reproduction, start):
    mo.stop(not start.value, mo.md("Kiểm tra cấu hình và GPU, rồi bấm **Chạy reproduction**."))
    results, output_dir = run_reproduction(config, fs)
    mo.vstack([
        mo.md(f"Kết quả đã sao lưu lên Google Drive: `{output_dir}`."),
        mo.ui.table(results),
    ])
    return

if __name__ == "__main__":
    app.run()
