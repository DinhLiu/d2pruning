# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "torch", "torchvision", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "tqdm"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _(mo):
    from gdrive_fsspec import GoogleDriveFileSystem

    fs = GoogleDriveFileSystem(use_listings_cache=False, skip_instance_cache=True, auth_kwargs={"use_local_webserver": False})
    mo.output.clear_console()
    return (fs,)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md("""
    # D² Pruning · CIFAR-10 · Chạy lại 70% sau sửa lỗi

    Đặt thư mục **cifar-10-batches-py/** cạnh notebook trong working directory.
    Gắn GPU từ **notebook specs** trước khi chạy. Dữ liệu được kiểm tra bằng
    torchvision với `download=False`; không tải lại CIFAR-10.

    Pipeline: khôi phục baseline/scores/features và log Random → chỉ chạy lại D² → CSV.
    Cố định seed 0, pruning 70%, 40.000 bước để so sánh trước/sau sửa lỗi.
    Chỉnh đường dẫn dữ liệu nếu cần. Log xuất hiện trong cell chạy.

    **Giới hạn:** hard cutoff `mis_ratio=0` là cấu hình thử nghiệm chưa được xác nhận
    từ bài báo. Không coi chênh lệch accuracy nhỏ là bằng chứng reproduce chính xác.
    Dùng commit bf1ea46b22cc79549df421fcd92c9cd2d3408008 và sửa lỗi căn chỉnh embeddings.
    Chỉ dành cho seed 0, pruning 70%, 40.000 bước của lần chạy đã xuất HTML.
    Giữ `mis_ratio=0` để đo riêng tác động của bản sửa; cutoff vẫn chưa được xác minh.
    Ưu tiên đọc output local cũ; nếu thiếu thì lấy từng file từ bản sao Google Drive cũ.
    Không huấn luyện lại baseline hoặc Random, không trích xuất lại features.

    Giữ cell kết nối Google Drive hiện có tạo biến **fs**. Cell chạy phụ thuộc vào fs.
    Notebook kiểm tra quyền ghi trước training, sao lưu vào `.ipynb_checkpoints/d2-<run-id>/`
    khi hoàn tất. Bản sao mới gồm CSV, checkpoint D², scores, features, indices,
    logs và bản vá mã nguồn. Checkpoint baseline gốc vẫn ở bản sao cũ.
    [Bài báo](https://arxiv.org/html/2310.07931v1) ·
    [Molab storage](https://docs.marimo.io/guides/molab/#storage)
    """)
    return


@app.cell
def _():
    config = {
        "data_path": "cifar-10-batches-py/",
        "output_path": "d2-reproduction-fixed70/",
        "previous_results_path": "d2-reproduction/results/",
        "previous_drive_path": ".ipynb_checkpoints/d2-20261023T171343863514Z/",
        "seeds": [0],
        "pruning_rates": [70],  # giữ nguyên cho lần kiểm tra này
        "steps": 40000,
        "mis_ratio": {30: 0.0, 50: 0.0, 70: 0.0, 80: 0.0, 90: 0.0},
        "methods": ["d2"],
        "drive_path": ".ipynb_checkpoints/",
    }
    # Giữ nguyên cấu hình cho lần kiểm tra lỗi này.
    # previous_drive_path là bản sao của lần chạy trong reproduce.html.
    # Nếu chạy lại từ đầu, đổi output_path; không ghi đè kết quả cũ.
    return (config,)


@app.cell
def _(mo):
    start = mo.ui.run_button(label="Chạy reproduction")
    start
    return (start,)


@app.function
def run_reproduction(config, fs):
    from pathlib import Path
    import os, sys, subprocess, json, shutil, re
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
    assert SEEDS == [0] and PRUNING_RATES == [70] and STEPS == 40000
    assert config['methods'] == ['d2'] and MIS_RATIO[70] == 0.0
    PREVIOUS = Path(config['previous_results_path']).expanduser().resolve()
    assert ROOT != PREVIOUS.parent, 'Dùng output_path mới để bảo toàn kết quả cũ'
    # Dừng nếu thư mục kết quả đã tồn tại: tránh dùng nhầm D² chưa sửa lỗi.
    assert not RESULTS.exists(), 'Output đã tồn tại; chọn output_path mới'
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
    def fix_feature_alignment(repo):
        path = repo / 'selection_mp.py'
        source = path.read_text()
        begin = source.index("    if args.coreset_mode == 'class':")
        end = source.index("    if args.coreset_mode == 'graph':", begin)
        block = source[begin:end]
        old = "            if mis_num:\n                features = features[score_index]"
        new = "            features = features[score_index]  # Align with scores and targets even at mis_num=0"
        if old in block:
            assert block.count(old) == 1
            block = block.replace(old, new)
        elif new not in block:
            raise RuntimeError('Không tìm thấy vị trí sửa căn chỉnh embeddings')
        path.write_text(source[:begin] + block + source[end:])

    fix_feature_alignment(REPO)
    (RESULTS / 'source-changes.patch').write_text(subprocess.check_output(
        ['git', 'diff', '--', 'selection_mp.py', 'train.py', 'core/data/MiscDataset.py',
         'generate_importance_score.py'], cwd=REPO, text=True))
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
                        torch=torch.__version__, torchvision=torchvision.__version__, sklearn=sklearn.__version__,
                        feature_alignment_fixed=True, methods=config['methods'],
                        previous_results_path=str(PREVIOUS), previous_drive_path=config['previous_drive_path'],
                        reused_tasks=['all-data', 'random-prune-70'])
    (RESULTS / 'manifest.json').write_text(json.dumps(manifest, indent=2))


    import pickle
    def restore_file(relative):
        destination = RESULTS / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        local = PREVIOUS / relative
        if local.is_file():
            shutil.copy2(local, destination)
            origin = str(local)
        else:
            remote = posixpath.join(config['previous_drive_path'].rstrip('/'), relative)
            # Dùng fs hiện có; thiếu file thì dừng, không tự huấn luyện baseline.
            with fs.open(remote, 'rb') as src, destination.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            origin = remote
        print('Dùng lại:', origin)
        return destination

    baseline_files = ['complete.json', 'data-score-all-data.pickle',
                      'train-features-conv3-all-data.npy', 'log-train-all-data.log']
    for name in baseline_files:
        restore_file('seed-0/all-data/' + name)
    for name in ['complete.json', 'log-train-random-prune-70.log']:
        restore_file('seed-0/random-prune-70/' + name)
    base = RESULTS / 'seed-0'
    baseline_meta = json.loads((base / 'all-data/complete.json').read_text())
    random_meta = json.loads((base / 'random-prune-70/complete.json').read_text())
    assert baseline_meta['commit'] == COMMIT and baseline_meta['steps'] == STEPS
    assert baseline_meta['seeds'] == [0] and baseline_meta['pruning_rates'] == [70]
    assert random_meta['seed'] == 0 and random_meta['steps'] == STEPS
    assert random_meta['method'] == 'random' and random_meta['pruning'] == 70
    for task, expected in [('all-data', 0.9539), ('random-prune-70', 0.9146)]:
        text = (base / task / f'log-train-{task}.log').read_text()
        best = re.findall(r'Best acc: ([0-9.]+)', text)
        assert best and abs(float(best[-1]) - expected) < 1e-8, 'Log không khớp reproduce.html'
    score = base / 'all-data/data-score-all-data.pickle'
    feature = base / 'all-data/train-features-conv3-all-data.npy'
    with score.open('rb') as f:
        scores = pickle.load(f)
    features = np.load(feature, mmap_mode='r')
    assert len(scores['forgetting']) == 50000 and features.shape == (50000, 512)
    assert len(scores['accumulated_margin']) == 50000
    assert np.isfinite(features).all()
    assert np.isfinite(np.asarray(scores['forgetting'])).all()
    assert np.isfinite(np.asarray(scores['accumulated_margin'])).all()
    baselines = {0: (base, score, feature)}
    print('Đã khôi phục baseline và Random; chỉ huấn luyện D² đã sửa lỗi.')


    PARAMS = {30:(10,0.9), 50:(5,1.0), 70:(1,0.1), 80:(2,0.0), 90:(2,0.0)}
    for seed, (base, score, feature) in baselines.items():
        for pruning in PRUNING_RATES:
            ratio = (100-pruning)/100
            k, gamma = PARAMS[pruning]
            assert 0 <= MIS_RATIO[pruning] < 1-ratio
            for method in config['methods']:
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
                counts = np.bincount(np.array(train_data.targets)[idx], minlength=10)
                assert np.all(counts == 1500), counts
                print('Class counts:', counts)
                (folder / 'complete.json').write_text(json.dumps(dict(seed=seed, method=method,
                    pruning=pruning, steps=STEPS, feature_alignment_fixed=True,
                    mis_ratio=MIS_RATIO[pruning] if method=='d2' else None)))


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
    fixed_acc = results.loc[results.method == 'd2', 'best_accuracy'].item()
    print(f'D² trước sửa: 90.33%; sau sửa: {fixed_acc:.2f}%; chênh lệch: {fixed_acc - 90.33:+.2f} điểm phần trăm')
    if len(SEEDS)>1:
        summary=results.groupby(['method','pruning']).agg(mean=('best_accuracy','mean'),std=('best_accuracy','std'),n=('seed','count'))
        print(summary.to_string())
        summary.to_csv(RESULTS/'summary.csv')
    print('Lưu toàn bộ output:', RESULTS)
    print('Kết quả D2 dùng cutoff thử nghiệm; chênh lệch số học không tự chứng minh tương đương thống kê.')

    backup()
    return results, remote_run


@app.cell
def _(config, fs, mo, start):
    mo.stop(not start.value, mo.md("Kiểm tra cấu hình và GPU, rồi bấm **Chạy reproduction**."))
    results, output_dir = run_reproduction(config, fs)
    mo.vstack([
        mo.md(f"Kết quả đã sao lưu lên Google Drive: `{output_dir}`."),
        mo.ui.table(results),
    ])
    return


if __name__ == "__main__":
    app.run()
