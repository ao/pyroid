# Real-hardware runs on AWS EC2 (`bench/ec2/`)

Native (no Docker) benchmark and parity runs on two EC2 instances, launched from your own machine and reached only
through SSM:

| arch | instance | CPU | why |
|---|---|---|---|
| x86_64 | `c7i.4xlarge`, 16 vCPU (8 cores × 2 threads), 32 GiB | Intel Xeon (Sapphire Rapids) | AVX2 + FMA: the x86_64 refs were pinned under Rosetta, which has no AVX2 |
| aarch64 | `c8g.4xlarge`, 16 vCPU (16 cores), 32 GiB | AWS Graviton4 (Neoverse V2) | aarch64 on server hardware |

Results are labelled with the instance, the CPU and the OS ("AWS EC2 c7i.4xlarge (Intel Xeon …, 16 vCPU), Amazon Linux
2023"): real hardware, but a VM. Never "bare metal".

| file | runs on | what |
|---|---|---|
| `ec2.sh` | your machine | every AWS call, each with `--profile $PYROID_AWS_PROFILE --region $PYROID_AWS_REGION` (default `us-west-2`): preflight, stage, launch, SSM, collect, terminate, S3 cleanup. Its header lists every setting |
| `user-data.sh` | the instance, at boot | first line `shutdown -h +420`: the instance powers off after 7 h and, launched with `--instance-initiated-shutdown-behavior terminate`, is terminated (root volume `DeleteOnTermination`) |
| `bootstrap.sh` | the instance, as root | dnf `git gcc gcc-c++ tar zstd`; rustup stable; uv 0.13.7; CPython 3.12.16, 3.13.18, 3.15.2, 3.15.2t (3.15 only builds the abi3t wheel, as in ci.yml); maturin 1.16.0. Idempotent |
| `run.sh` | the instance, as root | the run, as resumable steps (header comment): meta → code → wheels → envs → fetch → refs → parity → bench_dev → bench_headline → refs_hwcaps. Packs and uploads results after every step |
| `refdiff.py` | the instance (parity env) | `check`: regenerated refs vs the manifest, delta against the old pinned bytes, re-pin on the instance; `diff OLD NEW` |
| `summary.py` | your machine, after `collect` | Markdown tables (machines, step durations, parity, ref deltas, e2e, per stage) from the collected run folders |
| `run_warc.sh` | the instance, as root | the FineWeb-from-WARC benchmark (`../fineweb_warc.py`), on its own: see below |

## Safety
- Only `c7i.4xlarge`/`c8g.4xlarge`, at most 2 at a time; `--client-token` makes a repeated `launch` idempotent.
- One security group, `$PYROID_BENCH_SG` in `$PYROID_BENCH_VPC`, with no inbound rules (checked before every launch);
  no key pair, no SSH. Access is
  `ssm send-command` (IAM-authenticated). IMDSv2 only (`HttpTokens=required`, hop limit 1).
- Instance profile `pyroid-bench-profile`: SSM core plus Get/Put/List on `s3://$PYROID_BENCH_BUCKET/` only.
- `terminate` refuses any ID not in `/tmp/pyroid-ec2bench/instances.tsv` (launched here) and not tagged
  `project=pyroid-bench`, `Name=pyroid-bench-*`. Every launch and termination is a row in `$PYROID_EC2_LEDGER`
  (default `~/.pyroid-ec2bench/aws-resources.md`).
- Everything this writes to S3 is under `s3://$PYROID_BENCH_BUCKET/ec2bench/`; `ec2.sh s3-clean` removes it.

## Flow
```bash
E=bench/ec2/ec2.sh
$E preflight                          # read-only
$E stage                              # bundle $PYROID_BRANCH (+ overlay), scripts, old pins -> s3://…/ec2bench/
X=$($E launch x86 | tail -n 1); A=$($E launch arm | tail -n 1)
$E wait-ssm $X; $E wait-ssm $A
$E bootstrap $X; $E bootstrap $A      # foreground, ~10 min
$E start $X; $E start $A              # run.sh as systemd unit pyroid-run; returns at once
$E watch 300                          # status.json from S3 every 5 min until both are done/failed
$E collect x86; $E collect arm        # -> /tmp/pyroid-ec2bench/collect/<machine>/
$E terminate $X; $E terminate $A; $E verify; $E s3-clean
```
- **Code on the instance:** `$PYROID_BRANCH` (default `next`). If `$PYROID_OVERLAY_REF` names a second branch, the
  bench tooling (`bench/ec2`, `run_consolidated.sh`, `run_baselines.sh`, `consolidate.py`, `chart.py`,
  `machine_info.py`, `summarize.py`) is checked out from it (`code.txt` records both shas). Rebase that branch on
  `$PYROID_BRANCH` before `stage` so the overlay drops nothing, and pass the same variables to `ec2.sh start`.
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
a Linux container with Rust, uv, maturin and the CPythons that `bootstrap.sh` installs, and `bootstrap.sh` plus the build/env steps in an `amazonlinux:2023` container.

## FineWeb from WARC (`run_warc.sh`)
One c7i.4xlarge, separate from `ec2.sh`'s flow: the instance gets the script and `HTMLX_S3` (an `s3://` prefix with
`code/pyroid.bundle`, a git bundle of `HTMLX_BRANCH` [feat/trafilatura]) and runs it as root; its user data sets the
shutdown. The header of `run_warc.sh` lists every setting. Steps, most important first, with estimates for 8 WARC
files from P10's run (stock datatrove ~12 min at 8 workers, ~8x that at 1):

| step | min | what |
|---|---:|---|
| code, bootstrap, wheel, envs, fetch | ~22 | clone; `bootstrap.sh` (CPython 3.13.18 only); release wheel; `venv-stock` (`datatrove[io,processing]==0.10.1` + spaCy 3.8.18) and `venv-pyroid` (its freeze + the wheel); 8 WARC files of CC-MAIN-2026-39 and lid.176.bin |
| stock-8, pyroid-8, native-8 | 14, 4, 3 | `fineweb_warc.py` in each mode, 8 workers, `taskset -c 0-7` (one thread per core) |
| compare-8 | 1-4 | the three digests; if stock-8 lost pages to Trafilatura's timeout, `native-8x` (`--timeouts-of stock-8`) too |
| hook | ? | `HTMLX_HOOK` (or `HTMLX_HOOK_FILE`, or `$HTMLX_S3/code/hook.sh`): the parity scale runs |
| native-1, pyroid-1, stock-1 | 15, 17, 100 | the same at 1 worker |
| compare | 1-4 | every run against native-8 (+ `native-1x` if stock-1 lost pages) |

- Each step leaves `/opt/htmlx/state/<step>.done`; starting the script again skips finished steps, and
  `HTMLX_FORCE="stock-8"` redoes one. A run step whose estimate would end within 10 min of the scheduled shutdown is
  skipped, with the reason in `run.log`.
- After every step `/opt/htmlx/results/` (summaries, per-document hashes, timing records, datatrove's logs, the
  comparisons, `status.json`, `run.log`) is copied to `$HTMLX_S3/results/`. The run outputs stay on the instance.
- Needs about 15 GB free on the root volume (7.5 GB of WARC files, the Rust toolchain and build, two envs).
- Tested in the dev container with the setup steps stubbed (a local stand-in for `aws s3 cp`, the setup products
  linked in, `HTMLX_STOCK_ARGS="--lid-model PATH"` because the container cannot build fasttext-numpy2-wheel, and
  `HTMLX_RUN_ARGS` for small runs). The setup steps themselves follow P10's `research/htmlx/ec2/run.sh`, which ran
  on AL2023.
