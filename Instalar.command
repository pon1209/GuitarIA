#!/bin/bash
set -e
cd "$(dirname "$0")"
trap 'echo "Error de instalación. Revisa el mensaje anterior."; read -r' ERR
PY=/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11
if [ ! -x "$PY" ]; then
  echo 'Descargando Python 3.11 desde python.org…'
  PKG=$(mktemp -d)/python.pkg
  curl --fail --location --retry 2 https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg -o "$PKG"
  echo 'Completa el instalador de Python que se abrirá y vuelve a esta ventana.'
  open -W "$PKG"
  if [ ! -x "$PY" ]; then echo 'Python no está instalado. Vuelve a ejecutar este archivo tras instalarlo.'; read -r; exit 1; fi
fi
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
mkdir -p bin
.venv/bin/python - <<'PY'
import imageio_ffmpeg, pathlib, shutil, os
p=pathlib.Path('bin/ffmpeg'); shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(),p); p.chmod(0o755)
PY
./bin/ffmpeg -version | head -n 1
.venv/bin/python -c 'import torch, torchaudio, librosa, streamlit, demucs, basic_pitch; print("Dependencias importadas correctamente")'
echo 'Instalación terminada. Abre Iniciar.command.'
read -r
