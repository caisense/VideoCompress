# RB/1 重建时序修复记录

日期：2026-08-31

## 已实现

- 参考生命周期拆分为缓存 TTL `1000 ms`、内容硬时效 `450 ms`、future 窗 `100 ms` 和 STATE PTS 窗 `100 ms`。
- PATCH 在 FEC、CRC、JPEG 和 mask 校验通过后立即成为 PC 侧 `local-ready`，不再等待 ready STATE；STATE 只负责当前轨迹和几何。
- generation 与 reference_generation 使用单调代次检查；迟到的旧 STATE/PATCH、旧 SR candidate 和旧 SR result 不得覆盖新代次。
- Real-ESRGAN 保持一个 running 和一个 latest-only pending；提交前、完成后都检查 track、代次、内容年龄和 future 窗；所有 ROI 等比填充到固定 `96x96` 模型输入，避免 CUDA 动态形状重复准备。
- 同一个 H.265 `source_sequence` 只合成一次，并固定该次可用的 SR cache 视图；HOLD tick 复用完全相同的 composed pixels。
- 保留 bbox EMA、等比面积缩放、matchTemplate 内容配准和稳定的 `(generation, track_id, reference_generation)` SR key。
- sender 使用软刷新提示 `350 ms`、硬 deadline `450 ms`、guard `75 ms`；以参考采集时间计算内容年龄，按源帧量化机会调度，并结合参考大小、FEC、共享 RatePacer 和纯发送 P95 估算提前量。`350 ms` 不是从最后一个参考包发出后重新计时的最小保持。
- sender 记录 reference_generation、capture/encode/queue/first-send/last-send 时间、队列等待、采集到最后发送包的延迟、bytes、chunks、FEC 以及 P50/P95；这不是 PC 收包确认时间。
- rebuild 默认参考参数收敛为长边 `96`、JPEG 质量 `50`、JPEG 本体上限 `800 B`；典型 crop 使用一个 `1100 B` 数据分片，超大参考仍保留 XOR/FEC 路径。

## 兼容配置

`--rebuild-reference-soft-refresh-ms`、`--rebuild-reference-hard-deadline-ms` 和
`--rebuild-reference-refresh-guard-ms` 纳入 sender 配置；旧的
`--rebuild-patch-refresh-ms` 仍作为软阈值别名保留。

## RK3588 实机复验（2026-08-31）

- 使用 Ubuntu `192.168.0.10` 上的 `/opt/atk-dlrk3588-toolchain` 交叉编译，构建退出码为 `0`；安装包中的 sender 为 AArch64 ELF，RPATH 和 RK3588 运行库检查通过。
- 通过 staging 目录解包并原子替换开发板 `/opt/atk/rknn_yolov8_seg_cam`，旧目录保留为带日期的 backup；板端最终目录和二进制可执行，未触碰此前的旧 backup。
- 当前板端摄像头 `--max-frames=2` smoke 退出码为 `0`；随后使用板端 bus 测试视频跑最终 v4 默认 rebuild `60` 帧，退出码为 `0`，末帧统计为 `rebuild_refs=58`、`rebuild_patch_packets=58`、`rebuild_parity=0`、`rebuild_ref_chunks=1`、`tx_drop_p=0`、`tx_drop_idr=0`，参考 capture→last-send P95 为 `248.46 ms`，错误行计数为 `0`。
- 独立 PC UDP 计数器在同一板端运行中收到 `5004:24` 个 H.265 RTP 包和 `5009:54` 个 RB/1 包。HUD 实收 `profile=rebuild:256x144@6:g0`，`lost=0`、`reorder=0`、`decode_errors=0`，多帧 `READY=LOCAL MODE=ROI-LANCZOS`，PTS 同步约 `0/-1 ms`，稳定窗口观测约 `65-88 kbps`（100 kbps 组合上限内；启动突发单独统计）。
- 最终根因是 `scene_synced()` 的 future 判定符号写反：本应拒绝 `reference_pts - video_pts > 100 ms` 的未来参考，却错误拒绝了 `video_pts - reference_pts > 100 ms` 的历史参考。6 fps 下每代因此只可用 0/167 ms 两帧，333 ms 第三帧被清空，而异步 SR 恰好在其后才可消费。
- 修复后重新跑板端 bus 人物视频 120 帧。PC 实收 110 个 source frame（从 SEQ 11 开始），预热前 `ROI-LANCZOS=12`，预热后 `ROI-ESRGAN=98`，`BASE-LANCZOS=0`；其中 32 个年龄 `331-334 ms` 的第三帧均为 ESRGAN。CUDA warm-up `1780 ms`，运行 P95 不超过 `100 ms`，`lost/reorder/decode_errors=0/0/0`。
- 同段活动窗口 H.265+RB/1 物理速率平均 `57.7 kbps`、峰值 `86.7 kbps`，在 `100 kbps` 组合上限内。板端退出码 `0`，末帧 `rebuild_state=200`、`rebuild_refs=81`、`rebuild_patch_packets=81`、`tx_drop_p=0`、`tx_drop_idr=0`。

## 验证边界

- Python 全量回归：66/66；Python 语法检查和 `git diff --check` 通过。
- C++ 现有测试脚本的 11 个目标全部通过。
- 本机使用 `model\RealESRGAN_x2_dynamic.onnx` 和
  `CUDAExecutionProvider` 完成真实模型 smoke：`80x48 -> 160x96` 输出通过，后台
  warm-up 约 `1.8 s`；SR 固定输入形状、latest-only、future 延迟提交、精确参考代次、
  历史/未来 PTS 门限、过期前拒绝、完成后 stale 和同源帧像素不变测试均已覆盖。
- 板端摄像头画面本轮没有产生目标参考，因此参考生命周期的目标场景使用板端 bus 测试视频复验；尚未覆盖真实移动摄像头下的慢速/快速移动、遮挡和重新出现组合。

## goal0831 闭环复验（2026-08-31）

### debug 基线与修复结果

- `debug.txt` 基线共 389 个源帧：`BASE-LANCZOS=178`、`ROI-ESRGAN=137`、`ROI-LANCZOS=74`；RGEN 间隔 `p50=501 ms`、`p95=669 ms`，并存在最长 `3336 ms` 的异常间隔。
- 根因分别是 sender 用包发送完成时间做刷新保持、接收端把正常目标平移误判为中心越界，以及 SR 计算完成后仍受未来/同源帧显示门控影响。
- 当前 sender 以采集时间 deadline 调度；接收端拆分 `compute gate` 与 `render gate`，允许未来参考提前计算但不提前显示；几何闸门只拒绝无效、极端尺度/宽高、完全越界和低相关度结果。

### 最终实机结果

- Ubuntu `192.168.0.10` 交叉编译、checksum 校验和开发板原子替换均成功。最终包为
  `runs\goal0831_rk3588_final.tar.gz`，SHA-256 `638252aa3269e3d404811fa37cc0d145bbc7d0a1e123206847b6cd645c3e6f77`；板端二进制 SHA-256
  `8765b468592f3c5805e18ffddd1bc653bd923ee7b9801da7d51153ca2ddce05e`。
- 同一 bus 人物视频在板端运行 `180` 帧约 30 秒正常退出；PC 收到 164 个 source frame，`ROI-LANCZOS=88`、`ROI-ESRGAN=76`、`BASE-LANCZOS=0`，每帧 `REFREADY=1/1`、`USED=1/1`。
- 接收端 `AGEDROP/FUTUREDROP/NOREFDROP/STATEDROP/GENDROP/GEOMDROP/SCALEDROP/MATCHDROP=0`，`lost/reorder/decode_errors=0/0/0`；RGEN 间隔 `p50=334 ms`、`p95=335 ms`、最大 `337 ms`，SR 延迟 `p50=90 ms`、`p95=102 ms`。
- 完整日志：`runs\goal0831_sender_final.log`、`runs\goal0831_receiver_final.log`。

### 新增覆盖

- Python 全量回归由 66 项增至 69 项，补齐 moving-person 几何放行、不可行几何分因、内容 mismatch、future SR 预计算/缓存和 HUD 结构化字段；C++ refresh scheduler、慢发送和共享带宽测试全部通过。
- 真实 CUDA 模型路径在最终接收端实测出现 `ROI-ESRGAN`；同一 source sequence 仍保持像素不可变，防止后台结果回写当前已显示帧。
