from modal_env import app, triton_image, artifacts


@app.function(
    image=triton_image,
    gpu="A10G",
    timeout=600,
    volumes={"/artifacts": artifacts},
    env={
        "TRITON_ALWAYS_COMPILE": "1",
        "TRITON_KERNEL_DUMP": "1",
        "TRITON_DUMP_DIR": "/tmp/triton-dump",
    },
)
def run_pytest(test_path: str):
    import os
    import shutil
    import subprocess
    import tarfile
    import time

    import torch
    import triton

    dump_dir = "/tmp/triton-dump"

    shutil.rmtree(dump_dir, ignore_errors=True)
    os.makedirs(dump_dir)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__)
    print("Triton:", triton.__version__)

    subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-v",
            "-s",
            test_path,
        ],
        check=True,
    )

    # Triton 已经编译、执行完毕。
    # 此时再整理最终 artifact。
    run_id = time.strftime("%Y%m%d-%H%M%S")
    staging = f"/tmp/artifact-{run_id}"

    os.makedirs(staging)

    allowed = {
        ".ttir",
        ".ttgir",
        ".llir",
        ".ptx",
        ".cubin",
        ".sass",
        ".json",
    }

    for root, _, files in os.walk(dump_dir):
        for filename in files:
            if filename.startswith("tmp."):
                continue

            ext = os.path.splitext(filename)[1]

            if ext not in allowed:
                continue

            src = os.path.join(root, filename)

            # 保留 hash 目录等相对结构
            rel = os.path.relpath(src, dump_dir)
            dst = os.path.join(staging, rel)

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

            print("artifact:", rel)

    # 制作 immutable bundle
    archive = f"/tmp/triton-{run_id}.tar.gz"

    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staging, arcname="triton")

    # 最后才写入 Volume
    remote_dir = f"/artifacts/runs/{run_id}"
    os.makedirs(remote_dir, exist_ok=True)

    shutil.copy2(
        archive,
        f"{remote_dir}/triton-artifacts.tar.gz",
    )

    artifacts.commit()

    print(
        "Saved:",
        f"runs/{run_id}/triton-artifacts.tar.gz",
    )

    return run_id


@app.local_entrypoint()
def main(test_path: str = "tests/test_add.py"):
    run_id = run_pytest.remote(test_path)

    print()
    print("Run ID:", run_id)
    print(
        "Download with:\n"
        f"modal volume get triton-artifacts "
        f"runs/{run_id}/triton-artifacts.tar.gz"
    )

