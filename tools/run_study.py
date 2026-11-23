"""Repeat a completed 00–09 study, preserving its outputs for numerical comparison."""

from pathlib import Path
import hashlib, importlib.metadata, json, os, platform, shutil, subprocess, sys, time
from datetime import datetime, timezone
root = Path(__file__).resolve().parents[1]
required = [root / 'models', root / 'data/football_data_sources/download_manifest.csv']
required += [root / 'data' / name for name in ['MatchesChronological.csv', 'MatchTraining.csv', 'MarketTraining.csv']]
missing = [str(path.relative_to(root)) for path in required if not path.exists()]
if missing:
    raise SystemExit('Complete steps 00–09 before capturing a repeat run. Missing: ' + ', '.join(missing))
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup = Path('/tmp') / f'football-study-before-{stamp}'
out = root / 'reports' / 'reproduction'
out.mkdir(exist_ok=True)
backup.mkdir()
for folder in ['reports', 'docs', 'models']:
    shutil.copytree(root / folder, backup / folder, ignore=shutil.ignore_patterns('reproduction', '__pycache__'))
shutil.copy2(root / 'data/football_data_sources/download_manifest.csv', backup / 'download_manifest.csv')
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
paths = [root / 'data/Matches.csv', root.parent / 'outline.pdf']
paths += sorted((root / 'data/football_data_sources').glob('*.csv'))
paths += [p for folder in ['tests','tools'] for p in sorted((root / folder).glob('*.py'))]
paths += sorted(root.glob('*.py')) + [root / 'README.md', root / 'requirements.txt']
paths += [root / 'data' / name for name in ['MatchesChronological.csv','MatchTraining.csv','MarketTraining.csv']]
manifest = {'StartedAtUTC': datetime.now(timezone.utc).isoformat(), 'BackupDirectory': str(backup),
            'Python': sys.version, 'Platform': platform.platform(),
            'Packages': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','joblib','threadpoolctl']},
            'Command': '.venv/bin/python <step>', 'Steps': [],
            'InputAndInitialHashes': {str(p.relative_to(root)) if p.is_relative_to(root) else '../outline.pdf': digest(p) for p in paths}}
manifest_path = out / 'execution.json'
def save(): manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
save()
print(f'Backup: {backup}', flush=True)
env = os.environ.copy()
env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
steps = [(p.name, [sys.executable, p.name]) for p in sorted(root.glob('[0-9][0-9]_*.py'))]
steps += [('unit_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])]
for label, command in steps:
    started = time.monotonic()
    print(f'START {label}', flush=True)
    with (out / (label.removesuffix('.py') + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    record = {'Step':label, 'Command':command, 'ExitCode':result.returncode, 'Seconds':time.monotonic()-started}
    manifest['Steps'].append(record)
    save()
    print(f'END {label}: exit={result.returncode}, seconds={record["Seconds"]:.1f}', flush=True)
    print((out / (label.removesuffix('.py') + '.log')).read_text()[-2600:], flush=True)
    if result.returncode:
        sys.exit(result.returncode)
manifest['FinishedAtUTC'] = datetime.now(timezone.utc).isoformat()
save()
print('All steps and unit tests completed.', flush=True)
