# RB/1 重建时序修复记录

日期：2026-08-31

## 已实现

- 参考生命周期拆分为缓存 TTL `1000 ms`、内容硬时效 `450 ms`、future 窗 `100 ms` 和 STATE PTS 窗 `100 ms`。
- PATCH 在 FEC、CRC、JPEG 和 mask 校验通过后立即成为 PC 侧 `local-ready`，不再等待 ready STATE；STATE 只负责当前轨迹和几何。
- generation 与 reference_generation 使用单调代次检查；迟到的旧 STATE/PATCH、旧 SR candidate 和旧 SR result 不得覆盖新代次。
- Real-ESRGAN 保持一个 running 和一个 latest-only pending；提交前、完成后都检查 track、代次、内容年龄和 future 窗。
- 同一个 H.265 `source_sequence` 只合成一次，并固定该次可用的 SR cache 视图；HOLD tick 复用完全相同的 composed pixels。
- 保留 bbox EMA、等比面积缩放、matchTemplate 内容配准和稳定的 `(generation, track_id, reference_generation)` SR key。
- sender 使用软刷新阈值 `220 ms`、硬 deadline `450 ms`、guard `75 ms`，结合参考大小、FEC、共享 RatePacer 和观测 P95 传输时延调度刷新。
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
- PC 侧 `RealESRGAN_x2_dynamic.onnx` 的真实 CUDA provider/warm-up smoke 已通过；长流实机 HUD 日志出现 `loaded ROI model ... (CUDAExecutionProvider)`，warm-up 约 `4684 ms`，并在模型加载期间继续接收和显示 `ROI-LANCZOS`，`RUN/Q` 均未超过 `1`。由于低带宽流中参考代次持续滚动，本轮没有观察到 `ROI-ESRGAN` 显示命中，但 stale 结果被正确丢弃；不能把该结果表述为实机 SR 命中率验证。

## 验证边界

- Python RB/1 回归：31/31；HUD 回归：12/12；其余 4 个接收/协议脚本
  共 19/19；Python 语法检查通过。
- C++ 现有测试脚本的 11 个目标全部通过。
- 本机使用 `model\RealESRGAN_x2_dynamic.onnx` 和
  `CUDAExecutionProvider` 完成真实模型 smoke：`80x48 -> 160x96` 输出通过，后台
  warm-up 约 `1.8 s`；SR mock、latest-only、过期前拒绝、完成后 stale 和同源帧
  像素不变测试也已覆盖。
- 板端摄像头画面本轮没有产生目标参考，因此参考生命周期的目标场景使用板端 bus 测试视频复验；尚未覆盖真实移动摄像头下的慢速/快速移动、遮挡和重新出现组合。
