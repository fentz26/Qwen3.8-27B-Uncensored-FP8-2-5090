# Running vLLM as a systemd service

`scripts/serve.sh` runs vLLM as a bare backgrounded process — fine for
interactive setup, but it won't restart on crash and its logs go to a plain
file. `vllm-qwen.service` is the same exact configuration as a managed
systemd unit instead.

## Install

```sh
sudo cp vllm-qwen.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-qwen
```

## Operate

```sh
systemctl status vllm-qwen
journalctl -u vllm-qwen -f      # logs
sudo systemctl restart vllm-qwen
```

## Notes

- The unit's `ExecStart` mirrors `scripts/serve.sh` exactly (same flags —
  see README.md's "vLLM flags" table for what each one does and why). If
  you change one, change it in both places.
- Adjust the model path, venv path, and `HF_HOME` in the unit file if your
  layout differs.
- `TimeoutStartSec=900` — vLLM's first boot includes torch.compile and CUDA
  graph capture, which can take several minutes; systemd's default 90s
  startup timeout would kill it mid-compile otherwise.
