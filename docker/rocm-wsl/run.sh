#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

image="${WEATHER_RADAR_ML_IMAGE:-weather-radar-ml:rocm-wsl}"
dvc_remote="${WEATHER_RADAR_ML_DVC_REMOTE:-/mnt/nas}"
dockerfile="${script_dir}/Dockerfile"

build_image() {
    docker build \
        --file "${dockerfile}" \
        --tag "${image}" \
        "${repo_root}"
}

check_host() {
    local required_paths=(
        "/dev/dxg"
        "/usr/lib/wsl/lib/libdxcore.so"
        "/opt/rocm/lib/librocdxg.so"
        "/opt/rocm/share/rocdxg/dids.conf"
    )

    local path
    for path in "${required_paths[@]}"; do
        if [[ ! -e "${path}" ]]; then
            echo "Required WSL/ROCm path is missing: ${path}" >&2
            exit 1
        fi
    done

    if ! docker image inspect "${image}" >/dev/null 2>&1; then
        echo "Docker image '${image}' is missing." >&2
        echo "Run '$0 build' first." >&2
        exit 1
    fi
}

run_container() {
    check_host

    local docker_args=(
        --rm
        --user "$(id -u):$(id -g)"
        --device=/dev/dxg
        --cap-add=SYS_PTRACE
        --security-opt seccomp=unconfined
        --ipc=host
        --shm-size=8g
        --env HOME=/tmp
        --volume /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so:ro
        --volume /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so:ro
        --volume /opt/rocm/share/rocdxg/dids.conf:/usr/share/rocdxg/dids.conf:ro
        --volume "${repo_root}:/workspace/weather-radar-ml"
        --workdir /workspace/weather-radar-ml
    )

    if mountpoint -q "${dvc_remote}"; then
        docker_args+=(
            --volume "${dvc_remote}:${dvc_remote}"
        )
    fi

    if [[ -t 0 && -t 1 ]]; then
        docker_args+=("--interactive" "--tty")
    fi

    docker run "${docker_args[@]}" "${image}" "$@"
}

command="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${command}" in
    build)
        build_image
        ;;
    shell)
        run_container bash
        ;;
    gpu-check)
        run_container python -c \
            "import torch; print('PyTorch:', torch.__version__); print('ROCm:', torch.version.hip); print('GPU:', torch.cuda.get_device_name(0))"
        ;;
    test)
        run_container bash -lc \
            'python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))" && python -m pytest'
        ;;
    quality)
        run_container bash -lc \
            'python -m ruff format --check . && python -m ruff check . && python -m mypy'
        ;;
    run)
        if [[ $# -eq 0 ]]; then
            echo "Usage: $0 run COMMAND [ARGUMENTS...]" >&2
            exit 2
        fi
        run_container "$@"
        ;;
    help|-h|--help)
        echo "Usage: $0 {build|shell|gpu-check|test|quality|run}"
        ;;
    *)
        echo "Unknown command: ${command}" >&2
        echo "Usage: $0 {build|shell|gpu-check|test|quality|run}" >&2
        exit 2
        ;;
esac
