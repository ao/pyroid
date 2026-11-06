# Real-hardware runs on AWS EC2 (`bench/ec2/`)

Native (no Docker) benchmark and parity runs on two EC2 instances, launched from the Mac, reached only through SSM:

| arch | instance | CPU | why |
|---|---|---|---|
| x86_64 | `c7i.4xlarge`, 16 vCPU (8 cores × 2 threads), 32 GiB | Intel Xeon (Sapphire Rapids) | AVX2 + FMA: the x86_64 refs were pinned under Rosetta, which has no AVX2 |
| aarch64 | `c8g.4xlarge`, 16 vCPU (16 cores), 32 GiB | AWS Graviton4 (Neoverse V2) | aarch64 on server hardware |

Results are labelled with the instance, the CPU and the OS ("AWS EC2 c7i.4xlarge (Intel Xeon …, 16 vCPU), Amazon Linux
2023"): real hardware, but a VM. Never "bare metal".

| file | runs on | what |
|---|---|---|
| `ec2.sh` | the Mac | every AWS call (profile `primary1`, `us-west-2`, not overridable): preflight, stage, launch, SSM, collect, terminate, S3 cleanup |
| `user-data.sh` | the instance, at boot | first line `shutdown -h +420`: the instance powers off after 7 h and, launched with `--instance-initiated-shutdown-behavior terminate`, is terminated (root volume `DeleteOnTermination`) |
| `bootstrap.sh` | the instance, as root | dnf `git gcc gcc-c++ tar zstd`; rustup stable; uv 0.13.7; CPython 3.12.16, 3.13.18, 3.15.2, 3.15.2t (the pyroid-dev image's versions; 3.15 only builds the abi3t wheel, as in ci.yml); maturin 1.16.0. Idempotent |
| `run.sh` | the instance, as root | the run, as resumable steps (header comment): meta → code → wheels → envs → fetch → refs → parity → bench_dev → bench_headline → refs_hwcaps. Packs and uploads results after every step |
| `refdiff.py` | the instance (parity env) | `check`: regenerated refs vs the manifest, delta against the old pinned bytes, re-pin on the instance; `diff OLD NEW` |
| `summary.py` | Docker, after `collect` | Markdown tables (machines, step durations, parity, ref deltas, e2e, per stage) from the collected run folders |

## Safety
- Only `c7i.4xlarge`/`c8g.4xlarge`, at most 2 at a time; `--client-token` makes a repeated `launch` idempotent.
- Security group `sg-<redacted>` with no inbound rules (checked before every launch); no key pair, no SSH. Access is
  `ssm send-command` (IAM-authenticated). IMDSv2 only (`HttpTokens=required`, hop limit 1).
- Instance profile `pyroid-bench-profile`: SSM core plus Get/Put/List on `s3://$PYROID_BENCH_BUCKET/` only.
- `terminate` refuses any ID not in `/tmp/pyroid-ec2bench/instances.tsv` (launched here) and not tagged
  `project=pyroid-bench`, `Name=pyroid-bench-*`. Every launch and termination is a row in `.orchestration/aws-resources.md`.
- Everything this writes to S3 is under `s3://$PYROID_BENCH_BUCKET/ec2bench/`; `ec2.sh s3-clean` removes it.

## Flow
```bash
E=bench/ec2/ec2.sh
$E preflight                          # read-only
$E stage                              # bundle next + feat/ec2bench, scripts, old pins -> s3://…/ec2bench/
X=$($E launch x86 | tail -n 1); A=$($E launch arm | tail -n 1)
$E wait-ssm $X; $E wait-ssm $A
$E bootstrap $X; $E bootstrap $A      # foreground, ~10 min
$E start $X; $E start $A              # run.sh as systemd unit pyroid-run; returns at once
$E watch 300                          # status.json from S3 every 5 min until both are done/failed
$E collect x86; $E collect arm        # -> /tmp/pyroid-ec2bench/collect/<machine>/
$E terminate $X; $E terminate $A; $E verify; $E s3-clean
```
- **Code on the instance:** `next`, with the bench tooling (`bench/ec2`, `run_consolidated.sh`, `run_baselines.sh`,
  `consolidate.py`, `chart.py`, `machine_info.py`, `summarize.py`) checked out from `feat/ec2bench` when the bundle carries that
  branch (`code.txt` records both shas). Rebase `feat/ec2bench` on `next` before `stage` so the overlay drops nothing.
- **Time budget:** the 7-hour lifetime is a hard stop. Estimated on c7i: bootstrap + build + envs + fetch ~35 min, refs
  ~20, parity ~30, dev consolidated ~2.5 h, 100k headline ~1.5 h (1 run, no warmup: datatrove at 1 process is ~1 h per
  run). `bench_headline` estimates its time from the dev run's own medians and skips itself (with a reason in the log)
  when it would not finish 20 min before the shutdown; results are uploaded after every step and once more 15 min
  before the shutdown.
- **Re-pinning:** `refs` regenerates every reference natively. A mismatch with the manifest is quantified against the
  old pinned file (`ec2.sh stage` uploads `/tmp/pyroid-ec2bench/pins/`), then re-pinned on the instance so the parity
  suites and the bench gates run against this hardware's oracle outputs. `refs/manifest.diff` is what to commit;
  per-platform entries (`"platform"`) are expected to move, any other mismatch is flagged `unexpected`.

## Testing without AWS
`PYROID_S3` may be a local directory (same layout as the bucket) and `PYROID_DATA_SEED` a directory of already pinned
files (symlinked instead of downloaded; regenerated refs replace the links, never the targets). With small knobs
(`WHEEL_PROFILE=dev`, `REFS_WHAT=fasttext`, `PARITY_LIMIT=200`, tiny `BENCH_*` sizes) the whole of `run.sh` runs in
the pyroid-dev container, and `bootstrap.sh` plus the build/env steps in an `amazonlinux:2023` container.
