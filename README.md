# Pipeline trích xuất MP3 + Sub theo thuật ngữ FRP/Composite

Nhận link YouTube (video tiếng Việt có lẫn thuật ngữ chuyên ngành composite/FRP bằng
tiếng Anh) → transcribe bằng Whisper → tìm các đoạn có chứa thuật ngữ → xuất clip MP3
≤30s + phụ đề SRT + metadata (JSON/CSV) để dùng làm dataset cho đồ án/nghiên cứu.

## Luồng xử lý

```
YouTube URL
  → tải audio (yt-dlp, 16kHz mono WAV)
  → transcribe tiếng Việt (faster-whisper, tự chọn model theo VRAM GPU)
  → tìm thuật ngữ chuyên ngành (rapidfuzz, chịu được lỗi chính tả do ASR)
  → gộp thành clip ≤30s
  → cắt MP3 + ghi SRT (ffmpeg)
  → (tuỳ chọn) validate thêm bằng Claude Haiku qua OpenRouter
  → ghi metadata.json + metadata.csv + all_metadata.csv
```

## Yêu cầu hệ thống

- GPU NVIDIA (khuyến nghị ≥4GB VRAM). Không có GPU vẫn chạy được nhưng transcribe rất
  chậm (rơi về CPU).
- Python 3.11, ffmpeg.
- (Tuỳ chọn) API key OpenRouter nếu muốn bật bước validate thêm bằng Claude Haiku.

Chương trình **tự dò VRAM** (qua `nvidia-smi`) và tự chọn model Whisper phù hợp — không
cần biết trước máy có GPU gì:

| VRAM free | Model dùng |
|---|---|
| ≥ 8GB | `large-v3` + `float16` (chính xác nhất) |
| ≥ 4.5GB | `large-v3` + `int8_float16` |
| < 4.5GB | `medium` + `int8` (fallback an toàn cho GPU 4GB như laptop RTX 3050) |

Nếu gặp lỗi hết VRAM (CUDA OOM) giữa chừng, hệ thống tự động hạ cấu hình xuống mức an
toàn hơn và transcribe lại — không cần can thiệp thủ công.

---

## Cách 1: Cài trực tiếp bằng conda (chạy trên máy đang có sẵn Python/conda)

```bash
cd doan
conda env create -f environment.yml
conda activate doan-frp-pipeline
```

### Chạy bằng dòng lệnh (CLI)

```bash
# 1 video
python main.py "https://www.youtube.com/watch?v=xxxxxxxxxxx"

# Nhiều video cùng lúc — ghi danh sách link vào urls.txt (1 dòng/link), rồi:
python main.py --batch urls.txt

# Ghi đè thư mục output (mặc định lấy từ config.yaml)
python main.py "<url>" --output-dir /đường/dẫn/khác

# Bật thêm bước validate bằng Claude Haiku (cần API key OpenRouter)
python main.py "<url>" --openrouter-key sk-or-xxxxxxxx
# hoặc đặt biến môi trường thay vì truyền trên dòng lệnh:
export OPENROUTER_API_KEY=sk-or-xxxxxxxx
python main.py "<url>"
```

### Chạy bằng giao diện web (Streamlit) — dễ dùng hơn cho người không quen dòng lệnh

```bash
streamlit run app.py
```

Trình duyệt sẽ tự mở (hoặc vào `http://localhost:8501`). Giao diện có 3 ô:

1. **Thư mục output** — đường dẫn trên máy để lưu clip + metadata (tự tạo nếu chưa có).
2. **OpenRouter API key** — dán vào nếu muốn bật validate thêm bằng Claude Haiku (để
   trống nếu không cần, pipeline vẫn chạy bình thường).
3. **Link video YouTube** — dán link cần xử lý.

Bấm **Chạy**, chờ (video càng dài xử lý càng lâu), kết quả và log hiện ngay trên trang,
có thể nghe thử clip trực tiếp.

---

## Cách 2: Chạy bằng Docker (khuyến nghị khi chuyển sang máy khác có GPU mạnh hơn)

Trên máy đích (đã cài driver NVIDIA + Docker + [NVIDIA Container
Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```bash
cd doan
docker build -t doan-frp-pipeline .

docker run --gpus all -p 8501:8501 \
  -v $(pwd)/output:/app/output \
  doan-frp-pipeline
```

Sau đó mở `http://<ip-máy-đó>:8501` từ trình duyệt (cùng mạng LAN) — giao diện giống
hệt Cách 1. `-v $(pwd)/output:/app/output` để clip/metadata được lưu ra ngoài container,
không mất khi container dừng.

**Lưu ý**: image build lần đầu khá nặng (~1.4GB tải cuBLAS/cuDNN cho GPU) nên có thể mất
10-20 phút tuỳ tốc độ mạng, những lần build sau sẽ nhanh hơn nhờ cache.

---

## Cấu hình

### `terms.yaml` — danh sách thuật ngữ cần tìm

Thêm/bớt tự do, mỗi dòng 1 thuật ngữ (1 từ hoặc cụm nhiều từ):

```yaml
terms:
  - FRP
  - Filament Winding
  - Resin
  - Mandrel
  ...
```

### `config.yaml` — tinh chỉnh hành vi pipeline

Các mục quan trọng nhất:

- `whisper.model_size` / `whisper.compute_type`: để `auto` (khuyến nghị) hoặc ép cứng
  nếu muốn kiểm soát thủ công.
- `clip.max_duration_sec`: độ dài tối đa mỗi clip (mặc định 30s).
- `matching.fuzzy_threshold`: ngưỡng khớp gần đúng khi tìm thuật ngữ (mặc định 80/100).
  Giảm xuống nếu Whisper hay ghi sai chính tả thuật ngữ mà bị bỏ sót; tăng lên nếu bị
  match nhầm quá nhiều.

### `urls.txt` — danh sách link cho chế độ batch

Mỗi dòng 1 link, dòng bắt đầu bằng `#` bị bỏ qua.

---

## Kết quả đầu ra

```
output/
├── <video_id>/
│   ├── clips/
│   │   ├── clip_001.mp3
│   │   ├── clip_001.srt
│   │   └── ...
│   ├── metadata.json
│   └── metadata.csv
└── all_metadata.csv        # gộp tất cả video đã xử lý (batch mode)
```

Các cột trong CSV:

| Cột | Ý nghĩa |
|---|---|
| `video_id`, `video_title`, `source_url` | Thông tin video gốc |
| `clip_file`, `srt_file` | Đường dẫn tương đối tới clip/sub |
| `start_sec`, `end_sec`, `duration_sec` | Thời điểm cắt trong video gốc |
| `matched_terms` | Các thuật ngữ khớp được trong clip (cách nhau bằng `;`) |
| `confidence` | Độ tin cậy match (0-1, kết hợp xác suất ASR + độ khớp fuzzy) |
| `transcript_vi` | Transcript gốc từ Whisper (nguồn ASR thật, không bị sửa) |
| `llm_extra_terms` | Thuật ngữ Claude Haiku phát hiện thêm ngoài `terms.yaml` (nếu bật LLM) |
| `transcript_vi_llm_suggested` | Bản transcript Claude Haiku đề xuất sửa lỗi (nếu bật LLM) — **chỉ để đối chiếu, không ghi đè `transcript_vi`** |

---

## Lưu ý / giới hạn đã biết

- **Video càng dài xử lý càng lâu** — thời gian transcribe tỉ lệ gần tuyến tính với độ
  dài audio. VRAM dùng cho Whisper không tăng theo độ dài video (xử lý theo cửa sổ
  ~30s), nên video 1 tiếng không dễ OOM hơn video 3 phút, chỉ chậm hơn.
- Nếu CUDA OOM xảy ra **giữa chừng** lúc đang transcribe một video dài, hệ thống sẽ hạ
  cấu hình rồi **transcribe lại từ đầu** (không resume dở dang) — với video rất dài có
  thể mất thêm thời gian đáng kể nếu OOM xảy ra gần cuối.
- Giao diện Streamlit hiện chỉ hiện spinner khi đang chạy, chưa có thanh tiến trình chi
  tiết — với video dài có thể phải chờ khá lâu mà không thấy cập nhật liên tục.
- Bước validate bằng Claude Haiku gọi API tuần tự từng clip — video có nhiều clip khớp
  thuật ngữ sẽ mất thêm thời gian tương ứng số lần gọi API.
- yt-dlp thỉnh thoảng bị YouTube trả lỗi 403 tạm thời (rate-limit) — pipeline sẽ log lỗi
  và bỏ qua video đó chứ không crash cả batch; thử lại sau nếu gặp.
