# RDR-06 영상 인코더·타임라인 v0 — Frames -> H.264 mp4 (ffmpeg, libx264 / NVENC, 24 fps, 1080p/4K)

**목적.** S9-10 출력 `Frames`(FND-02)를 검증 가능한 mp4로 만든다. 프레임 수·fps·타임스탬프·색이 계약대로이고, 실패해도 반쯤 쓴 파일을 남기지 않는다. 정형 래퍼다: 영리한 것을 넣지 않는다.
**소유.** `buc/render/video.py`, 테스트 `tests/rdr_06/`(동결). 의존: FND-02 `buc.core.contracts.Frames`, numpy, 표준 라이브러리, 시스템 `ffmpeg`/`ffprobe`. **torch 금지**(CPU import 가능, `import buc.render.video` 후 `sys.modules`에 torch 없음). `buc/render/__init__.py`는 만들지 않는다(namespace).

## 공개 API (`from buc.render import video`)
- `class VideoError(RuntimeError)` — ffmpeg 없음·인코더 없음·프로세스 실패·닫힌 writer 사용. 잘못된 인자는 `ValueError`/`TypeError`(VideoError 아님).
- `resolution(name) -> (W,H)`: `"1080p"`->(1920,1080), `"4k"`->(3840,2160), 대소문자·앞뒤 공백 무시, 그 외 `ValueError`.
- `fps_rational(fps) -> Fraction`: `Fraction(float(fps)).limit_denominator(1001)`. 유한·`0<fps<=1000` 아니면 `ValueError`. (24000/1001 -> 24000/1001)
- `frame_count(duration_s, fps) -> int` = `int(round(duration_s*fps))`(음수 duration·잘못된 fps `ValueError`). `frame_times(n, fps) -> float64 (n,)` = `arange(n)/fps`(n<0 `ValueError`, n=0 빈 배열).
- `to_uint8(x) -> uint8 C-연속`: uint8은 그대로; float32/float64는 유한하고 `[-1e-4, 1+1e-4]` 안이어야 하며(밖/NaN/inf `ValueError`) `rint(clip(x,0,1)*255)`(반올림 짝수). 그 외 dtype `TypeError`. 임의 ndim, 입력 불변(읽기 전용 허용).
- `find_ffmpeg(name="ffmpeg") -> str|None`(`shutil.which`).
- `probe_encoder(encoder, *, ffmpeg=None) -> bool`: 256x256 1프레임을 `-f null -`로 실제 인코딩해 본다(목록에 있어도 GPU가 없으면 False). 예외 없이 bool, 프로세스 내 캐시.
- `select_encoder(codec="h264", prefer="auto", *, available=None, ffmpeg=None) -> str`: codec∈{h264,hevc}, prefer∈{auto,cpu,gpu}(아니면 `ValueError`). 후보 h264=(`h264_nvenc`,`libx264`), hevc=(`hevc_nvenc`,`libx265`); auto=GPU 우선 후 CPU, cpu=CPU만, gpu=GPU만. `available`(이름 집합)이 주어지면 프로브 없이 거기에 있는 것만 사용, 없으면 `probe_encoder`. 쓸 수 있는 게 없으면 `VideoError`(gpu 요청 시 조용한 CPU 폴백 금지).
- `build_ffmpeg_cmd(out_path, *, width, height, fps: Fraction, encoder, crf=18, preset="medium", ffmpeg="ffmpeg") -> list[str]` 순수(셸 문자열 금지, 전부 str). 입력 옵션(`-i -` 앞): `-f rawvideo -pix_fmt rgb24 -s WxH -r num/den`. 출력 옵션: `-an -map_metadata -1 -fflags +bitexact -flags:v +bitexact`, `-vf scale=out_color_matrix=bt709:out_range=tv`, `-pix_fmt yuv420p`, `-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv`, `-c:v ENC`, `-movflags +faststart`, 마지막 원소 = out_path; `-hide_banner -loglevel error -y` 포함. libx264/libx265: `-crf C -preset P`(P=x264 이름 9종). NVENC: `-rc vbr -cq C -b:v 0 -preset p1..p7`(ultrafast..veryslow -> p1,p2,p3,p3,p4,p5,p6,p7,p7), `-crf` 없음. libx265/hevc_nvenc는 `-tag:v hvc1`. `ValueError`: W/H가 <2·홀수, crf∉[0,51], 모르는 preset/encoder, fps<=0.
- `@dataclass(frozen=True) EncodeResult(path: Path, n_frames, width, height, fps: float, codec, encoder, bytes, cmd: tuple[str,...])`; `VideoInfo(codec, width, height, fps: float, n_frames, duration_s, pix_fmt, color_space, color_range)`.
- `class VideoWriter(path, *, width, height, fps, codec="h264", encoder="auto", crf=18, preset="medium", overwrite=True, ffmpeg=None)`: `encoder`는 `"auto"|"cpu"|"gpu"` 또는 ffmpeg 인코더 이름. 확장자 `.mp4|.mov|.mkv`만(아니면 `ValueError`). 생성자에서 인자 검증(홀수 크기·fps·crf 등)하고 실패 시 아무것도 만들지 않는다. `overwrite=False`이고 파일이 있으면 `FileExistsError`. 상위 디렉터리 자동 생성. 인코딩은 같은 디렉터리의 임시 파트 파일(`<이름>.part<확장자>`)로 하고 성공 시에만 `os.replace`(원자적). 속성 `n_frames`, `result`(close 후 EncodeResult). `write(frames)`: `(H,W,3)` 또는 `(k,H,W,3)`, uint8 또는 float [0,1](`to_uint8` 규칙), 크기 불일치 `ValueError`; 프레임은 즉시 ffmpeg stdin으로 흘려보낸다(4K 전체를 메모리에 쌓지 않는다). `close() -> EncodeResult`는 멱등, 프레임 0개면 `ValueError`(파일·파트 없음), ffmpeg 실패면 `VideoError`(파트 삭제). close 후 `write`는 `VideoError`. 컨텍스트 매니저: 본문에서 예외가 나면 프로세스를 죽이고 파트 파일을 지우며 **기존 `path`는 건드리지 않고** 예외를 그대로 올린다.
- `encode_frames(frames: Frames | ndarray(N,H,W,3), path, *, fps=None, codec, encoder, crf, preset, overwrite, ffmpeg) -> EncodeResult`: `Frames`면 `frames.fps` 사용(다른 `fps` 인자는 `ValueError`), ndarray는 `fps` 필수, ndim≠4 `ValueError`. 내부는 `VideoWriter`.
- `probe(path, *, ffprobe=None) -> VideoInfo`(ffprobe `-count_frames`; 파일 없음 `FileNotFoundError`, 읽을 수 없는 파일·바이너리 없음 `VideoError`). `decode_frames(path, *, ffmpeg=None) -> uint8 (N,H,W,3)` rgb24(태그된 색 행렬 사용), 실패 `VideoError`.
- `concat_frames(clips: Sequence[Frames]) -> Frames`: 비면 `ValueError`; (H,W)·fps가 모두 같아야 함(아니면 `ValueError`); rgb는 순서대로 이어 붙임; aov는 **모든 클립에 있는 키만** 첫 축으로 연결; `shot_hash`/`world_hash`는 전 클립이 같을 때만 유지, 아니면 `""`; 입력 불변.

## 불변식
1. 출력 프레임 수 = 입력 N, 24 fps면 `r_frame_rate=24/1`, 프레임 i의 PTS = i/fps(허용 1e-4 s), 길이 = N/fps. fps는 `fps_rational`로 `num/den` 문자열로 전달(24000/1001 보존).
2. 색: `Frames.rgb`는 표시값(Rec.709 원색)이므로 RGB->YUV는 **BT.709 제한 범위**(Y=16+219·Y', Cb/Cr=128+224·…), 컨테이너 태그도 bt709/tv. 순색 7종의 YUV가 공식과 ±2 이내.
3. 결정론: 같은 입력·파라미터·같은 ffmpeg 빌드의 libx264 출력은 **바이트 동일**(파일명·시각 무관). NVENC는 보장하지 않는다. float 입력과 `to_uint8` 한 uint8 입력은 같은 바이트.
4. 원자성: 실패·예외 후 디렉터리에 최종 파일도 파트 파일도 남지 않고, 기존 파일은 그대로다.
5. 인코더 선택은 프로브 기반: nvenc가 ffmpeg 목록에 있어도 GPU가 없으면 auto는 libx264로 간다.

## 수용 기준
`tests/rdr_06/` 전부 통과(CPU, 약 5 s, `-m "not gpu"`): 순수 함수(해상도·fps·타임라인·uint8·선택·명령행 레이아웃), libx264 왕복(프레임 수·fps·PTS·색 태그·PSNR>=36 dB), 색 행렬 독립 검증, 결정론, 스트리밍=일괄, 1080p·4K, 원자성·검증 오류, concat. NVENC 1080p/4K는 `@pytest.mark.gpu`(GPU 러너, nvenc 불가면 skip). 동결 해시 `FND-02`와 같은 방식(`RDR-06.tests.json`).

## 하류 이음매
EVL-02(일일 A/B 러너)는 `encode_frames(frames, path)`로 24장 고정 무브를 mp4로 만들고 `probe`로 프레임 수·fps를 확인한다. PRD-01/S9-10 산출 `Frames`가 입력이다. 샷 이어붙이기는 `concat_frames`. 해상도 이름은 `resolution`. NVENC 사용은 GPU 러너에서 `encoder="gpu"` 또는 `"auto"`. 영상 바이트의 비트 동일성 주장(RDR-09)은 Frames 해시 기준이며 NVENC 영상에는 걸지 않는다. 8K·HEVC 10-bit·ProRes/EXR은 범위 밖(HEVC 8-bit는 선택만 지원).
**소유 밖 요청.** 없음(`ffmpeg`/`ffprobe`는 시스템 패키지 전제).
