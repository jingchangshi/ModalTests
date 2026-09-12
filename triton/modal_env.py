import modal

app = modal.App("triton-tests")

artifacts = modal.Volume.from_name(
    "triton-artifacts",
    create_if_missing=True,
)

triton_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch",
        "triton",
        "numpy",
        "pytest",
    )
    .add_local_python_source(
        "modal_env",
        "kernels",
        "tests",
    )
)
