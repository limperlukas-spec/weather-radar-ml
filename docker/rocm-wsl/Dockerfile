FROM rocm/pytorch:rocm7.14_ubuntu24.04_py3.11_pytorch_release_2.12.0

RUN python -m pip install --no-cache-dir \
    "hydra-core==1.3.7" \
    "mlflow==3.16.1" \
    "numpy==2.4.6" \
    "xarray==2026.7.0" \
    "zarr==3.1.6" \
    "build==1.5.0" \
    "dvc==3.67.1" \
    "mypy==1.20.2" \
    "mutmut==3.8.0" \
    "pytest==8.4.2" \
    "pytest-cov==6.3.0" \
    "ruff==0.16.8"

WORKDIR /workspace/weather-radar-ml

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --no-cache-dir --no-deps --editable .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace/weather-radar-ml/src

CMD ["bash"]
