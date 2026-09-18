# Environment records

`reported_environment.yaml` is a manuscript transcription, not a recovered
Conda lock file. It deliberately records the unresolved Ultralytics version
conflict. `Dockerfile.cpu` is only a CPU build-and-test smoke environment; it
does not reproduce GPU throughput or paper training time.

Capture the actual machine before every run:

```bash
python scripts/system_info.py --output run_environment.json
```
