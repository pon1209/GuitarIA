#!/bin/bash
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ] || [ ! -x bin/ffmpeg ]; then echo 'Ejecuta Instalar.command primero.'; read -r; exit 1; fi
export PATH="$PWD/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
