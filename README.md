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

- **Bắt buộc GPU NVIDIA ≥ 8GB VRAM.** Không hỗ trợ GPU <8GB hoặc chạy CPU — thực nghiệm
  cho thấy GPU yếu (vd 4GB) buộc phải hạ xuống model nhỏ + beam_size=1, gây lỗi
  **timestamp không khớp audio** (transcript lệch so với audio thật trong clip). Chương
  trình sẽ báo lỗi rõ ràng và dừng lại nếu VRAM không đủ, thay vì âm thầm chạy ở mức kém
  tin cậy.
- Python 3.11, ffmpeg.
- (Tuỳ chọn) API key OpenRouter nếu muốn bật bước validate thêm bằng Claude Haiku.

Chương trình **tự dò VRAM** (qua `nvidia-smi`) và luôn dùng model `large-v3` (không còn
model nhỏ hơn) — chỉ khác nhau ở compute_type:

| VRAM free | Model dùng |
|---|---|
| ≥ 10GB (GPU thực tế ≥12GB, vd RTX 3080 10GB/4070 Ti 12GB/3060 12GB) | `large-v3` + `float16` (tốt nhất) |
| ≥ 6.5GB (GPU thực tế ~8GB, vd RTX 3070/3080) | `large-v3` + `int8_float16` |
| < 6.5GB | **Báo lỗi và dừng** — không còn fallback xuống model nhỏ hơn |

Nếu gặp lỗi hết VRAM (CUDA OOM) giữa chừng ở tier `float16`, hệ thống tự động hạ xuống
`int8_float16` và transcribe lại; nếu `int8_float16` cũng OOM thì báo lỗi (không còn
tier thấp hơn để hạ xuống nữa).

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

### Deploy lên GPU cloud thuê (RunPod, Vast.ai, ...)

Đã test thực tế trên GPU server thuê ngoài (RTX 5060 Ti 16GB) — pipeline chạy đúng, có
1 vấn đề cần biết:

**yt-dlp bị lỗi `403 Forbidden` khi tải video trên hầu hết server cloud/datacenter.**
Nguyên nhân THẬT (đã xác nhận bằng test trực tiếp): **thiếu JS runtime để giải
signature/n-parameter challenge của YouTube** — không phải do IP bị chặn hẳn. Fix bằng
cách cài [Deno](https://deno.land) (`curl -fsSL https://deno.land/install.sh | sh`) và
đảm bảo nó có trong `PATH`. **Dockerfile đã tự cài Deno**, nên dùng Cách 2 (Docker) để
deploy RunPod thì không cần làm gì thêm.

Nếu deploy không qua Docker (vd RunPod serverless custom worker), nhớ cài Deno thủ công
trong quá trình build worker image.

Nếu vẫn bị chặn dai dẳng dù đã có Deno (YouTube có thể tăng cường chặn theo thời gian):
đặt biến môi trường `YTDLP_COOKIES_FILE` trỏ tới file `cookies.txt` xuất từ trình duyệt
đã đăng nhập YouTube (dùng extension "Get cookies.txt LOCALLY"), `downloader.py` sẽ tự
dùng file này. `main.py` cũng tự thử lại với nhiều `player_client` khác nhau
(`android_vr`, `tv`, `web`) trước khi báo lỗi.

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
- `clip.max_duration_sec` / `clip.min_duration_sec`: độ dài clip (mặc định 5-30s). Clip
  được căn theo **ranh giới câu hoàn chỉnh** của Whisper (không cắt cụt giữa câu), mở
  rộng sang câu liền kề nếu ngắn hơn `min_duration_sec`.
- `matching.fuzzy_threshold`: ngưỡng khớp gần đúng khi tìm thuật ngữ (mặc định 80/100).
  Giảm xuống nếu Whisper hay ghi sai chính tả thuật ngữ mà bị bỏ sót; tăng lên nếu bị
  match nhầm quá nhiều.
- `llm_validation.confidence_threshold`: ngưỡng % tự tin (mặc định 95) để cột `llm_valid`
  đánh nhãn `TRUE`/`FALSE` (chỉ có tác dụng khi bật API key OpenRouter).

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
| `llm_confidence_rate` | % tự tin (0-100) của Claude Haiku rằng transcript **mạch lạc về ngữ nghĩa** và thuật ngữ dùng đúng ngữ cảnh — **chỉ đánh giá dựa trên văn bản, không nghe lại audio gốc** nên không phát hiện được trường hợp Whisper transcribe sai hoàn toàn nhưng câu vẫn nghe "trôi chảy" |
| `llm_valid` | `TRUE` nếu `llm_confidence_rate` ≥ ngưỡng cấu hình (mặc định 95%), ngược lại `FALSE` |

---

## Lưu ý / giới hạn đã biết

- **Bắt buộc GPU ≥ 8GB.** Chương trình từ chối chạy trên GPU yếu hơn (báo lỗi rõ ràng
  `InsufficientVRAMError`) — xem phần "Yêu cầu hệ thống" phía trên để biết lý do.
- **Video càng dài xử lý càng lâu** — thời gian transcribe tỉ lệ gần tuyến tính với độ
  dài audio. VRAM dùng cho Whisper không tăng theo độ dài video (xử lý theo cửa sổ
  ~30s), nên video 1 tiếng không dễ OOM hơn video 3 phút, chỉ chậm hơn.
- Nếu CUDA OOM xảy ra **giữa chừng** lúc đang transcribe một video dài, hệ thống sẽ hạ
  từ `float16` xuống `int8_float16` rồi **transcribe lại từ đầu** (không resume dở
  dang) — với video rất dài có thể mất thêm thời gian đáng kể nếu OOM xảy ra gần cuối.
  Nếu `int8_float16` cũng OOM, video đó bị bỏ qua (báo lỗi, không crash cả batch).
- **Timestamp Whisper vẫn có thể trôi (audio không khớp SRT) ngay cả ở GPU đủ mạnh** —
  đây là giới hạn xác suất của ASR nói chung, không riêng gì tier thấp. Clip đã được cắt
  theo đúng ranh giới câu hoàn chỉnh (không cắt cụt), nhưng **chưa có bước verify lại
  bằng cách transcribe ngược audio đã cắt** để tự động phát hiện/loại bỏ clip bị lệch —
  đây là điểm cần cải thiện tiếp theo nếu cần độ tin cậy cao hơn cho dataset.
- Cột `llm_confidence_rate`/`llm_valid` chỉ đánh giá dựa trên **văn bản** transcript
  (mạch lạc, đúng ngữ cảnh) — **không nghe lại audio gốc**, nên không phát hiện được
  trường hợp Whisper transcribe sai hoàn toàn nhưng câu vẫn nghe hợp lý (chính là dạng
  lỗi mô tả ở mục trên). Đừng dùng cột này như bằng chứng audio khớp SRT.
- Giao diện Streamlit hiện chỉ hiện spinner khi đang chạy, chưa có thanh tiến trình chi
  tiết — với video dài có thể phải chờ khá lâu mà không thấy cập nhật liên tục.
- Bước validate bằng Claude Haiku gọi API tuần tự từng clip — video có nhiều clip khớp
  thuật ngữ sẽ mất thêm thời gian tương ứng số lần gọi API.
- yt-dlp thỉnh thoảng bị YouTube trả lỗi 403 tạm thời (rate-limit) — pipeline sẽ log lỗi
  và bỏ qua video đó chứ không crash cả batch; thử lại sau nếu gặp.
