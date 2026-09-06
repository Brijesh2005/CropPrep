import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

arr = json.load(open('artifacts/r5_5_pull/logs_v5.json', encoding='utf-8'))
print('total lines:', len(arr))
keys = ['checkpoint', '=== Phase', 'binary variant', '[training] epoch', 'diagnose_exit', 'wrote report', 'complete', 'baselines_return', 'Traceback', 'RuntimeError', 'missing', 'ERROR', 'probe_batch', 'tiny', 'overfit', 'report ->']
for e in arr:
    d = e.get('data', '')
    if any(k in d for k in keys):
        t = e.get('time', 0)
        print("[%8.2fs] %s: %s" % (t, e.get('stream_name','?'), d), end='')