# Running on ClearML

How a run gets from this checkout onto a GPU agent, and the traps between here and there. For what each parameter means once it is running, see [configurations](configurations/README.md).

## The shape of a run

```
make run                    # PYTHONPATH=. uv run src/train.py
  -> init_clearml()         # creates the task, stamps the docker image on it
  -> config_clearml()       # connects every parameter group to the UI
  -> task.execute_remotely()  # ENDS the local process; the task is now a draft
```

Nothing trains locally. `execute_remotely()` kills the process on the next line, leaving a task in draft state that you then enqueue — from the ClearML UI, or with `Task.enqueue(task, queue_name=...)`. The agent that picks it up clones this repository at the commit recorded on the task, pulls the docker image recorded on the task, and runs `src/train.py` inside it.

Three consequences follow, and each has bitten this project at least once.

**A value typed into the ClearML UI only ever reaches the agent.** The local launcher dies before reading it. The exceptions are `clearml_project` and `clearml_task_name`, which are read before `Task.connect()` and so only apply to a locally launched first run — an existing task keeps the project it was created in, and a clone inherits it. Move or rename an existing task with the UI's own fields.

**The commit must be reachable by the agent.** The task pins the commit it was created from, so an unpushed branch fails at clone time. `version_num` on an existing task also wins over its branch name: editing the branch alone does not move the task to newer code, you have to clear the commit for it to follow the branch. Agents cache clones per repository, and a stale cache can write an old branch name back onto the task after execution.

**The image must exist on the agent's host.** With `docker_force_pull: false` a locally built image is used as-is, which means an image built on one host is invisible to agents on another. Either build on each host or push to a registry with `make build push TRAINER_IMAGE=ghcr.io/acme/yolo-trainer:0.2.10`.

## Images and versioning

`./VERSION` is the only place the version is written; `src/params.py` turns it into `DOCKER_IMAGE`, the Makefile reads that constant back out, and `set_base_docker()` stamps it onto every task. `make build` and what the agents pull therefore cannot drift.

**Never re-tag a published version.** A task stores the image tag it was created with and keeps requesting it forever, so overwriting `yolo-trainer:0.2.0` silently changes what every past task reruns on. `make bump PART=patch` before any build whose image contents changed.

**The image shadows `src/`.** The container sets `PYTHONPATH=/workspace` and bakes a copy of `src/` in, so `src.data.*` and `src.yolov8.*` resolve to the **image's** copy while `src/train.py` runs from the git checkout. Editing anything under `src/` has no effect on a remote run until the image is rebuilt. The two can drift silently.

**Anything that varies per release goes at the bottom of the Dockerfile.** A changed `--build-arg` invalidates every instruction below the one that consumes it, and `IMAGE_VERSION` feeds a `LABEL`. With that block at the top, a version bump alone rebuilt `apt-get` (71s) and `uv sync` (352s): 495s for a five-byte change. Moving the `ARG`/`LABEL` below the `COPY` steps took the same bump to 4s.

**Keep the version out of `pyproject.toml` and `uv.lock`.** Both are bind-mounted into the `uv sync` layer and BuildKit keys that layer on their contents, so a one-line change there re-installs about 8GB of dependencies. `pyproject.toml` declares `dynamic = ["version"]`, which uv accepts for a virtual project — nothing here is ever built or published — and `tests/utils/test_version.py` fails if a static version returns to either file.

## Inside the container

**It runs as root on purpose.** clearml-agent's docker-mode bootstrap writes to `/etc/apt`, `/root/.cache/pip` and `/root/.ssh` and runs `apt-get`. Adding a `USER` directive to the Dockerfile makes every remote task hang before reaching `src/train.py`.

**`set_base_docker()` replaces the whole container section.** Passing `docker_image` without `docker_arguments` drops the `CLEARML_AGENT_SKIP_*` env vars, and the agent then ignores the image's baked venv and rebuilds one with pip. Always pass both.

**`--ipc=host` and `--shm-size=50gb` are not optional.** Dataloader workers exchange batches through shared memory, and Docker's 64MB default shows up as a silent worker crash mid-epoch rather than as an error about shared memory.

**`torch` is pinned to a CUDA 12.x wheel window** (`>=2.9,<2.10`). Newer torch resolves to a CUDA 13 wheel that needs a much newer driver and, if unmet, silently falls back to CPU instead of erroring. Check every agent host's driver before widening it.

**Python 3.14 defaults `multiprocessing` to `forkserver` on Linux**, not `fork`. Forkserver re-imports `__main__` in every dataloader worker, so anything at module scope in `src/train.py` would re-run per worker — re-initialising ClearML and re-downloading datasets. The `if __name__ == "__main__":` guard is what prevents that; keep all work inside `main()`.

## Logging on an agent

The agent's console is the one that gets read, so the volume of it is a design constraint rather than a preference.

| Level | Meaning |
|---|---|
| ERROR | the run is affected: an export failed, validation crashed |
| WARNING | degraded but continuing: CVAT export timed out, unpaired labels |
| INFO | stage boundaries, and one summary line per unit of work |
| DEBUG | per-item detail: each filter decision, raw response bodies |

**The one rule: no INFO line may be emitted from inside a loop over dataset items.** Tally with `src.utils.logging.Tally` and log one summary. Log volume must not grow with dataset size, and `tests/data/test_data_stage_smoke.py` asserts exactly that by running each stage at two input sizes and expecting the same line count.

Turn the volume up with `LOG_LEVEL=debug` as an env var, or with the `0_Console` group in the UI — see [0_console](configurations/0_console.md) for the two caveats about when each one applies. Measuring against the baseline of 334 lines of our own output on a 3-epoch, 4-CVAT-task run: `grep -cE '\| src\.' console.txt`.

**Quieten ultralytics per call, never with `YOLO_VERBOSE=0`.** The env var is read once in `ultralytics/utils/__init__.py` and drops every ultralytics logger to ERROR — the per-epoch metrics table and the AMP/dataset warnings go with it. `args_val["verbose"]` and `args_predict["model"]["verbose"]` are the right knobs, and `src/train.py` turns them back on at `LOG_LEVEL=DEBUG`.

**Console parameters live in `args_console`, not `args_logging`.** `config_clearml()` does `args_train.update(args_logging)` and `args_train` is splatted into `model_yolo.train()`, which rejects unknown keys outright: `SyntaxError: 'log_level' is not a valid YOLO argument`.
