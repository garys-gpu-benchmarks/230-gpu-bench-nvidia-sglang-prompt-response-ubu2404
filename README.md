# SGLang Prompt-Response Benchmark

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/garys-gpu-benchmarks/230-gpu-bench-nvidia-sglang-prompt-response-ubu2404/actions/workflows/ci.yml/badge.svg)](https://github.com/garys-gpu-benchmarks/230-gpu-bench-nvidia-sglang-prompt-response-ubu2404/actions/workflows/ci.yml)

Measures sequential SGLang prompt-response latency. Workload 230 of the NVIDIA Ubuntu 24.04 bundle.

## Quick Start

On an Ubuntu 24.04 host, as root or with sudo:

```bash
git clone https://github.com/garys-gpu-benchmarks/230-gpu-bench-nvidia-sglang-prompt-response-ubu2404.git
cd 230-gpu-bench-nvidia-sglang-prompt-response-ubu2404
sudo bash setup.sh --assume-yes
bash run_benchmark.sh --smoke
```

`run_benchmark.sh` runs setup by itself if it has not been run yet. Each run prints its metrics when it finishes.

For this LLM serving workload, `--baseline` and `--extended` start the real model server, wait for it to be ready, run the client, and stop the server.

The model `mistralai/Mistral-7B-v0.3` may be gated on Hugging Face. If its download is refused, set `HF_TOKEN` (see [Troubleshooting](#troubleshooting)).

## Profiles and Parameters

| Parameter | Flag | smoke | baseline | extended |
|---|---|---|---|---|
| `dtype` | `--dtype` | bfloat16 | bfloat16 | bfloat16 |
| `input_len` | `--input-len` | 64 | 512 | 1024 |
| `output_len` | `--output-len` | 16 | 860 | 8100 |
| `model_name` | `--model-name` | mistralai/Mistral-7B-v0.3 | mistralai/Mistral-7B-v0.3 | mistralai/Mistral-7B-v0.3 |
| `num_prompts` | `--num-prompts` | 2 | 8 | 8 |
| `max_running_requests` | `--max-running-requests` | 1 | 1 | 1 |
| `max_total_tokens` | `--max-total-tokens` | 1024 | 16384 | 32768 |
| `seed` | `--seed` | 42 | 42 | 42 |

Pick a profile with `--smoke` (default), `--baseline` or `--extended`. Any flag in the table overrides that profile's value:

```bash
bash run_benchmark.sh --baseline --num-prompts 4
bash run_benchmark.sh --help        # every option
```

Profile values live in `config/benchmark_config.yaml`.

## Metrics

| # | Metric | Key |
|---|---|---|
| 1 | E2E latency p50, ms | `end_to_end_latency_ms` |
| 2 | Time To First Token p50, ms | `ttft_ms` |
| 3 | Output token generation rate (decode) p50, tokens/s | `output_tokens_per_s` |
| 4 | RadixAttn hit rate, pct | `radixattention_cache_hit_rate_pct` |
| 5 | Completed request rate, pct | `completed_request_rate_pct` |

### Notes

- The smoke profile is a quick check with reduced sizes and iteration counts. Use `--baseline` or `--extended` for representative numbers.
- Port 30000 is exclusive with 231.
- Every profile starts real SGLang (about 2 min startup).
- Ready-wait up to 1800 s.
- Sequential client, not concurrent serving.

## Output

| Path | Contents |
|---|---|
| `results/raw/YYYYMMDD_HHMMSSZ_WORKLOAD-<NNN>_SYSTEM-<hostname>/` | Raw tool output, `metrics_summary.txt`, and `hardware_info.txt`, `software_info.txt`, `errors_info.txt` |
| `results/benchmark.db` | SQLite history of every run |
| `results/summary.json` | Metrics from the most recent run |
| `/var/opt/benchmarks/runtime_ledger.csv` | One row per run, shared by every workload on the host |

Every run is checked for completeness and against any `thresholds:` or `baselines:` in `config/benchmark_config.yaml`. A run that fails the checks exits non-zero.

## Requirements

- Ubuntu 24.04, root or sudo for `setup.sh`
- NVIDIA GPU with CUDA 12.6
- CUDA Runtime, PyTorch-CUDA, Hugging Face Transformers, Mistral-7B-v0.3, SGLang, HTTP client harness, Python 3.12. `setup.sh` installs what is missing; Python packages go into a repo-local `.venv`.
- A Hugging Face token with access to `mistralai/Mistral-7B-v0.3`

## Run the Whole Bundle

This workload is also part of [`bundle-nvidia-ubuntu-2404`](https://github.com/garys-gpu-benchmarks/bundle-nvidia-ubuntu-2404), which holds all 32 NVIDIA Ubuntu 24.04 workloads as submodules:

```bash
git clone --recurse-submodules https://github.com/garys-gpu-benchmarks/bundle-nvidia-ubuntu-2404 /opt/benchmarks
cd /opt/benchmarks
./run_benchmark_suite.sh -w 230 --smoke     # just this workload
./run_benchmark_suite.sh --smoke            # all 32 workloads
./show_metrics.sh 230                       # this workload's runs from the ledger
```

The suite tools (`run_benchmark_suite.sh`, `get_benchmark_results.sh`, `list_runs.sh`, `show_metrics.sh`) are documented in [`gpu-bench-suite`](https://github.com/garys-gpu-benchmarks/gpu-bench-suite).

Other platforms: [AMD 24.04](https://github.com/garys-gpu-benchmarks/bundle-amd-ubuntu-2404) (101–132), [AMD 26.04](https://github.com/garys-gpu-benchmarks/bundle-amd-ubuntu-2604) (301–332), [NVIDIA 26.04](https://github.com/garys-gpu-benchmarks/bundle-nvidia-ubuntu-2604) (401–432).

## Troubleshooting

- **SSH dropped during setup:** reconnect and run `sudo bash setup.sh --assume-yes` again. It resumes; do not delete `.venv` or `.cache`.
- **Model download refused (401 or 403):** `mistralai/Mistral-7B-v0.3` may be gated. Accept its license on Hugging Face, then `export HF_TOKEN=<token>` and run setup with `sudo -E bash setup.sh --assume-yes` so the token reaches it.
- **A metric reads `na`:** the run could not measure it on this host. The reason is in the run's `metrics_summary.txt` and `errors_info.txt`; see also [Notes](#notes).

## Repository Layout

```text
.
├── setup.sh                      # installs dependencies and builds the benchmark
├── run_benchmark.sh              # runs one profile, parses, validates
├── show_metrics.sh               # prints this workload's ledger rows
├── SPEC.md                       # engineering specification
├── benchmark_specification.json  # workload definition
├── config/                       # profile values and thresholds
├── scripts/                      # build, collect, parse, validate
├── docs/                         # PRD and contracts
├── tests/
├── results/                      # run outputs
└── LICENSE
```

## Continuous Integration

- [CI](.github/workflows/ci.yml) runs lint and static checks on every pull request and push to `main`. It does not run the benchmark.
- [GPU Smoke Benchmark](.github/workflows/gpu-smoke.yml) runs a profile on a self-hosted runner labeled `gpu`, `nvidia`, `ubu2404`. It only starts by hand (**Actions → GPU Smoke Benchmark → Run workflow**), so pull requests and forks cannot run it.

Both workflows call shared steps in [`shared-workflows@v1`](https://github.com/garys-gpu-benchmarks/shared-workflows).
