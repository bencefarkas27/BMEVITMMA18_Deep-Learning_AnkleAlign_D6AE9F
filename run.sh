#!/usr/bin/env bash
# run.sh - run the full pipeline scripts in order
# This script is used by the Docker image and local testing to execute the
# main pipeline stages in sequence for demonstration purposes.

set -euo pipefail

URL="https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQB8kDcLEuTqQphHx7pv4Cw5AW7XMJp5MUbwortTASU223A?e=Uu6CTj&download=1&xsdata=MDV8MDJ8fDIyOTc1YmYyMWMzNzQyODFlZWZhMDhkZTM3YmNkMjdifDZhMzU0OGFiNzU3MDQyNzE5MWE4NThkYTAwNjk3MDI5fDB8MHw2MzkwMDk0ODEyNTc5MDU5MTR8VW5rbm93bnxWR1ZoYlhOVFpXTjFjbWwwZVZObGNuWnBZMlY4ZXlKRFFTSTZJbFJsWVcxelgwRlVVRk5sY25acFkyVmZVMUJQVEU5R0lpd2lWaUk2SWpBdU1DNHdNREF3SWl3aVVDSTZJbGRwYmpNeUlpd2lRVTRpT2lKUGRHaGxjaUlzSWxkVUlqb3hNWDA5fDF8TDNSbFlXMXpMekU1T2xSM1NIcHViVlpTVlVKUGFGUjFRVTFyWlc1blIyVlhTSEkzYjB0WVNWQkNTamxxTWtKbkxVdFdkMnN4UUhSb2NtVmhaQzUwWVdOMk1pOWphR0Z1Ym1Wc2N5OHhPVHBVZDBoNmJtMVdVbFZDVDJoVWRVRk5hMlZ1WjBkbFYwaHlOMjlMV0VsUVFrbzVhakpDWnkxTFZuZHJNVUIwYUhKbFlXUXVkR0ZqZGpJdmJXVnpjMkZuWlhNdk1UYzJOVE0xTVRNeU5ETTJPQT09fDBiYmVmZWIwYWJmOTRkZTFlZWZhMDhkZTM3YmNkMjdifGRlNDNhNjEyMWZmNzQxOTk4OGJiYzk4ZWMzZjU4MTdk&sdata=SWRDUWQrSVVCZTViZ05ZVEp2dU10ZFhJWG90RTdYZCtxSjBXbEtUclBCND0%3D&ovuser=6a3548ab-7570-4271-91a8-58da00697029%2Cbencefarkas27%40edu.bme.hu"

OUTPUT="data/anklealign.zip"
EXTRACT_DIR="data"

wget --no-check-certificate --content-disposition -O "$OUTPUT" "$URL"

if [ $? -eq 0 ]; then
    echo "Download completed successfully"
    unzip -o "$OUTPUT" -d "$EXTRACT_DIR"
else
    echo "Download failed"
    exit 1
fi

echo "[run.sh] Starting full pipeline run at $(date --iso-8601=seconds)"

echo "[run.sh] Running data preprocessing at $(date --iso-8601=seconds)"
python src/01_data_preprocessing.py
echo "[run.sh] Running training at $(date --iso-8601=seconds)"
python src/02_train.py
echo "[run.sh] Running evaluation at $(date --iso-8601=seconds)"
python src/03_evaluation.py
echo "[run.sh] Running inference at $(date --iso-8601=seconds)"
python src/04_inference.py

echo "[run.sh] Pipeline finished at $(date --iso-8601=seconds)"