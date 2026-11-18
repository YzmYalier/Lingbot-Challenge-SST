#!/usr/bin/env bash
# Make torchcodec find an FFmpeg 7 runtime inside the conda env.
#
# Why this is needed: torchcodec ships libtorchcodec_*4/5/6/7.so but no FFmpeg
# itself; it dlopen()s libavutil.so.{56,57,58,59} from the system at import
# time. Our dev box has ubuntu's ffmpeg 4 (libavutil.so.56) in /usr/lib, so
# torchcodec's v4 libs load and video decoding works. The cluster task image has
# no FFmpeg at all, so every candidate fails and the DataLoader dies with
# "Could not load libtorchcodec ... libavutil.so.59: cannot open shared object
# file". It is a missing OS library on the cluster, not a broken conda env.
#
# The fix reuses the complete FFmpeg 7 runtime that the `av` (PyAV) wheel already
# vendors in site-packages/av.libs. Those files are named with a build hash
# (libavutil-a63ffd27.so.59.39.100), so the dynamic loader never finds them by
# soname. We create a shim directory of soname symlinks and bake it into the
# RPATH of torchcodec's v7 libraries.
#
# RPATH (rather than LD_LIBRARY_PATH in train.sh) is deliberate: it is scoped to
# these three .so files only. Putting av.libs on LD_LIBRARY_PATH would expose
# ~45 vendored libs (libX11, libdrm, libvulkan, libssl, ...) to every process in
# the job, which risks shadowing the system copies that RoboTwin's renderer and
# CUDA use.
#
# Idempotent: safe to re-run. Re-run after reinstalling torchcodec or av.
set -euo pipefail

PYTHON="${PYTHON:-python}"

# The patch is written into the conda env, so once applied it persists on shared
# storage and every later job inherits it. Exit early in that case, before
# requiring patchelf: the tool is needed to apply the fix, not to use it, and the
# cluster task image does not necessarily have it.
if "${PYTHON}" - 2>/dev/null <<'PY'
import ctypes, os.path, sys, torch, torchcodec  # torch first: libtorch.so is a NEEDED dep
lib = os.path.join(os.path.dirname(torchcodec.__file__), "libtorchcodec_core7.so")
try:
    ctypes.CDLL(lib)
except OSError:
    sys.exit(1)
PY
then
  echo "torchcodec already resolves an FFmpeg 7 runtime; nothing to do."
  exit 0
fi

command -v patchelf >/dev/null 2>&1 || {
  echo "ERROR: patchelf not found. Install it (apt-get install -y patchelf, or pip install patchelf)." >&2
  exit 1
}

read -r TC_DIR AV_DIR <<EOF
$("${PYTHON}" - <<'PY'
import os.path, torchcodec, av
print(os.path.dirname(torchcodec.__file__),
      os.path.join(os.path.dirname(os.path.dirname(av.__file__)), "av.libs"))
PY
)
EOF

[ -d "${AV_DIR}" ] || { echo "ERROR: no vendored FFmpeg at ${AV_DIR}. Is 'av' pip-installed?" >&2; exit 1; }

SHIM_DIR="${TC_DIR}/.ffmpeg_shim"
mkdir -p "${SHIM_DIR}"

# FFmpeg 7 sonames, matching what libtorchcodec_core7.so lists as NEEDED.
for spec in avutil:59 avcodec:61 avformat:61 avdevice:61 avfilter:10 swscale:8 swresample:5; do
  name="${spec%%:*}"
  major="${spec##*:}"
  src="$(ls -1 "${AV_DIR}/lib${name}-"*".so.${major}."* 2>/dev/null | head -1 || true)"
  if [ -z "${src}" ]; then
    echo "ERROR: lib${name}.so.${major} not found in ${AV_DIR}." >&2
    echo "       The vendored 'av' wheel is not FFmpeg 7; check 'pip show av'." >&2
    exit 1
  fi
  ln -sfn "${src}" "${SHIM_DIR}/lib${name}.so.${major}"
done

# $ORIGIN keeps this relocatable: the shim resolves the FFmpeg sonames, and
# av.libs resolves FFmpeg's own vendored deps (libdrm, libvpx, libx264, ...).
#
# --force-rpath writes DT_RPATH instead of DT_RUNPATH, and that matters here.
# DT_RUNPATH applies only to the object that carries it, so libavutil.so.59 --
# which has no runpath of its own -- would fail to find libdrm-b0291a67.so.2.4.0
# and the whole load would abort. DT_RPATH is inherited down the dependency
# chain, so the vendored transitive deps resolve too.
RPATH='$ORIGIN/.ffmpeg_shim:$ORIGIN/../av.libs'
for lib in libtorchcodec_core7.so libtorchcodec_custom_ops7.so libtorchcodec_pybind_ops7.so; do
  [ -f "${TC_DIR}/${lib}" ] || { echo "ERROR: missing ${TC_DIR}/${lib}" >&2; exit 1; }
  patchelf --force-rpath --set-rpath "${RPATH}" "${TC_DIR}/${lib}"
done

# Prove it decodes, not just that it links. torchcodec probes versions 7->4 and
# swallows per-version failures, so an import alone would pass even if v7 were
# still broken and a system ffmpeg were picked up instead.
"${PYTHON}" - <<'PY'
import ctypes, os.path, torch, torchcodec  # torch first: libtorch.so is a NEEDED dep
lib = os.path.join(os.path.dirname(torchcodec.__file__), "libtorchcodec_core7.so")
ctypes.CDLL(lib)  # raises OSError if the FFmpeg 7 sonames still do not resolve
print("OK: FFmpeg 7 runtime resolved for torchcodec", torchcodec.__version__)
PY

echo "Done. Shim: ${SHIM_DIR}"
