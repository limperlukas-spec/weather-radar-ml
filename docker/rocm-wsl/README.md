# AMD ROCm on WSL2

This directory provides the hardware-specific development environment for running
Weather Radar ML on an AMD GPU through WSL2 and Docker.

The setup has been validated with:

- Windows with WSL2 and Ubuntu 24.04
- AMD Radeon RX 7900 XTX (`gfx1100`)
- ROCm 7.14 with `rocdxg-roct`
- Docker Engine running inside WSL2
- PyTorch 2.12.0 with ROCm 7.14

## Compatibility note

The canonical Python environment currently declares `torch>=2.14.0`. The official
ROCm 7.14 container used here provides PyTorch 2.12.0. The project is therefore
installed with `--no-deps` so that the container-provided ROCm build is preserved.

The complete test suite passes in this environment, but this remains an explicit
platform compatibility override until an appropriate PyTorch 2.14 ROCm image is
available or the project's minimum supported PyTorch version is revised.

The Dockerfile pins the directly installed dependencies to the versions used for
the validated test run. Transitive dependencies are still resolved by pip during
a clean image build.

## Prerequisites

The WSL2 host must provide:

```text
/dev/dxg
/usr/lib/wsl/lib/libdxcore.so
/opt/rocm/lib/librocdxg.so
/opt/rocm/share/rocdxg/dids.conf
```

Docker must be usable by the current WSL user without `sudo`.

## Commands

Build the image:

```bash
./docker/rocm-wsl/run.sh build
```

Verify GPU access:

```bash
./docker/rocm-wsl/run.sh gpu-check
```

Run formatting, linting, and type checking:

```bash
./docker/rocm-wsl/run.sh quality
```

Run the complete test suite:

```bash
./docker/rocm-wsl/run.sh test
```

Open an interactive container shell:

```bash
./docker/rocm-wsl/run.sh shell
```

Run an arbitrary command:

```bash
./docker/rocm-wsl/run.sh run python --version
```

The repository is bind-mounted into the container. Files created by the container
use the invoking WSL user's UID and GID.

## DVC remote on the NAS

The project-specific NAS share is mapped in Windows as:

```text
N: -> \\TRUENAS\weather-radar-ml-dvc
```

WSL mounts that drive at `/mnt/nas`. The local `/etc/fstab` entry is:

```text
N: /mnt/nas drvfs defaults,nofail 0 0
```

This host-specific configuration is not part of the repository. The mount can be
checked with:

```bash
mountpoint /mnt/nas
```

The container launcher bind-mounts the DVC remote when that mount point is active.
Configure the machine-local DVC remote with:

```bash
./docker/rocm-wsl/run.sh run \
  dvc remote add --local --force --default nas /mnt/nas
```

DVC writes this setting to `.dvc/config.local`, which is ignored by Git.

For the 0.7 benchmark, retrieve the prepared ML dataset and the 0.6 reference
artifacts with:

```bash
./docker/rocm-wsl/run.sh run dvc pull \
  data/ml/radklim-yw-2023-09-dortmund-128x128-60m-30m-v1-1906c133.dvc \
  data/reference/radklim-yw-0.6-reference.dvc
```

## Environment overrides

Use a different local image name:

```bash
WEATHER_RADAR_ML_IMAGE=my-image:tag \
  ./docker/rocm-wsl/run.sh build
```

Use a different mounted DVC remote:

```bash
WEATHER_RADAR_ML_DVC_REMOTE=/another/mount \
  ./docker/rocm-wsl/run.sh run dvc remote list
```
