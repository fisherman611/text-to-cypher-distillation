# text-to-cypher-distillation

## 1. Cài môi trường bằng `uv`

Yêu cầu:

- Python >= 3.10
- Bash
- GPU/CUDA nếu train

Cài `uv`:

```bash
pip install uv
```

Cài dependencies:

```bash
uv sync
```

Kích hoạt môi trường:

```bash
source .venv/bin/activate
```

Nếu bạn dùng Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Lưu ý:

- Repo đang khóa `torch==2.8.0+cu128` trong `pyproject.toml`.
- Nếu máy bạn không dùng CUDA 12.8, có thể cần sửa lại version `torch` trước khi `uv sync`.

## 2. Chạy từng file trong `updated_span_scripts/`

Tất cả script nằm trong:

```text
updated_span_scripts/qwen/
```

Hiện tại có các nhóm:

- `updated_span_scripts/qwen/kd`
- `updated_span_scripts/qwen/sfkl`
- `updated_span_scripts/qwen/csd`
- `updated_span_scripts/qwen/distillm`
- `updated_span_scripts/qwen/fdd`

Mỗi file `.sh` là một cấu hình train hoàn chỉnh. Bạn chạy trực tiếp file đó bằng `bash`.

Ví dụ:

```bash
bash updated_span_scripts/qwen/sfkl/train_0.6B_4B_sfkl_kd0.7_wrel1.0.sh
```

Hoặc:

```bash
bash updated_span_scripts/qwen/kd/train_0.6B_4B_rkl_kd0.7_wrel1.0.sh
```

### Các script này đang làm gì

Mỗi script thường:

- tự set `CUDA_VISIBLE_DEVICES`
- gọi `torchrun`
- chạy `updated_finetune.py`
- lưu checkpoint vào `results/qwen3/<ten_script>/`

Ví dụ file `updated_span_scripts/qwen/sfkl/train_0.6B_4B_sfkl_kd0.7_wrel1.0.sh` sẽ lưu vào:

```text
results/qwen3/sfkl_train_0.6B_4B_sfkl_kd0.7_wrel1.0/
```

### Chọn GPU

Mặc định, nếu không truyền gì, script thường sẽ dùng `GPUS=(0 1)`.

Nếu muốn đổi GPU, truyền biến môi trường `RUN_GPUS`:

```bash
RUN_GPUS=2,3 bash updated_span_scripts/qwen/sfkl/train_0.6B_4B_sfkl_kd0.7_wrel1.0.sh
```

### Sửa cấu hình train

Nếu muốn đổi dataset, model, teacher model, learning rate, output path..., sửa trực tiếp trong file `.sh` cần dùng. Những biến thường cần sửa:

- `DATA_DIR`
- `CKPT`
- `TEACHER_CKPT`
- `SAVE_PATH`
- `LR`
- `BATCH_SIZE`
- `EPOCHS`

## 3. Chạy `running.sh`

`running.sh` dùng để quét tất cả file `.sh` trong `updated_span_scripts/qwen` và tự động xếp lịch chạy.

Xem hướng dẫn:

```bash
bash running.sh --help
```

Chạy toàn bộ script:

```bash
bash running.sh
```

Chỉ chạy những script có chữ `sfkl` trong đường dẫn:

```bash
bash running.sh --filter sfkl
```

Chỉ chạy những script có chữ `kd0.7_wrel1.0` và dùng 4 GPU, mỗi job 2 GPU:

```bash
bash running.sh --filter kd0.7_wrel1.0 --gpus 0,1,2,3 --gpus-per-job 2
```

Chạy lần lượt từng script:

```bash
bash running.sh --mode sequential
```

Chạy thử mà không launch job:

```bash
bash running.sh --dry-run
```

Log mặc định:

```text
run_logs/<timestamp>/
```

Trong đó thường có:

- `run.log`
- `success.log`
- `failed.log`
- `retry.log`
- `jobs/*.log`

## 4. Lệnh dùng nhanh

```bash
uv sync
source .venv/bin/activate
bash running.sh --filter distillm
bash running_2.sh --filter distillm_new_3
```
