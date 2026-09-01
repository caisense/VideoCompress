# VideoCompress Face-Aware Bit Allocation 实施与实机报告

日期：2026-09-01  
项目：`D:/workspace/videoCompressV1`  
基线：`401c9bc`

## 1. 基线与目标

本次实现保留现有 PTS、状态机、播放时序、参考帧和 SR 交接逻辑，只增加参考图像的目标区域选择、JPEG 质量策略、协议标记、遥测和离线分析。默认模式仍为 `person`，只有 A/B 通过后才允许改为 `adaptive`。

原问题是人物框占据较大面积，而固定 800 B 的参考 JPEG 需要把整个人物框压缩到很低的源像素密度；HUD 中虽然有 `ROI-ESRGAN`，但它只能放大已收到的像素，不能恢复被编码阶段丢掉的脸部细节。此次 HEAD 模式把预算集中到上半部代理区域，验证这一判断。

## 2. 根因验证

- 旧路径以完整 person bbox 作为参考图，人物框中包含大量背景和非关键区域，有限字节被摊薄。
- 100 kbps 是 V+RB+音频的上限预算，不等于每个参考 JPEG 都能得到更多有效脸部像素。
- `ROI-ESRGAN` 的触发和结果显示属于接收端渲染阶段，不能修复发送端 JPEG 的低源分辨率。
- 选择器没有引入新的人脸神经网络；HEAD 是基于 person bbox、分割 mask 上半 55% 和 proxy crop 的可解释代理。

## 3. HEAD 选择器

实现位于 `cpp/transport/rebuild_reference.h/.cc`。有有效 mask 时使用上半 55% 的前景；无 mask 时使用 bbox 顶部到 `0.45H` 的 proxy。两者都应用 12% margin、最小尺寸 24、边界裁剪和可见区域检查。微小框、无效框和非 person 类回退 FULL，并记录 `HEAD_FALLBACK=SMALL` 等原因。

形式化选择器测试覆盖 mask、tiny、edge clamp、non-person、坐姿/部分目标和无 mask proxy。测试通过。

## 4. JPEG 与预算策略

- FULL：目标 Q50，最低 Q40，单个参考 JPEG 不超过 800 B。
- HEAD：目标 Q60，最低 Q45，单个参考 JPEG 不超过 800 B。
- 两种模式都保持现有参考发送节奏和 450 ms 硬截止，不增加第二代或额外 cadence。
- 本次实机采样中，FULL 实际 Q 的 p50/p95 为 40/40；HEAD 为 45/45，说明 800 B 上限最终约束了质量档位。

## 5. 协议与接收端

继续使用协议 v1 和原 40-byte PATCH metadata。`left/top/right/bottom` 表示实际发送 crop，`reference_left/top/right/bottom` 保留完整 detector bbox。packet flags 使用 bit0 parity 和 bit1 `0x0002` HEAD，旧接收端可忽略新 bit。

接收端按完整 bbox 与实际 crop 的坐标关系映射回当前帧，HEAD 不再把小 crop 拉伸成完整 detector bbox；没有 HEAD 标记时保持原 FULL 行为。协议组包、flag 一致性和精确坐标映射均有回归测试。

## 6. 遥测与调试样本

发送端和 HUD 已记录参考 kind、source crop 尺寸、JPEG 尺寸/Q/bytes/scale、head pixels、linear/area gain、p50/p95 统计和 fallback。`--rebuild-debug-reference-dir` 打开时输出 source、JPEG、selector；接收端输出 decoded、mask、SR、final。关闭时不产生 I/O。

两组正式实机运行共生成 158 个 selector 样本（person 95、adaptive 63），并直接检查了代表性样本。样本显示 adaptive 在本次现场画面选择到了举起的手和上半部代理区域，而不是稳定的人脸区域，因此不能把本次视觉结果宣称为“人脸识别正确”。

## 7. 自动化测试

在 Ubuntu 交叉编译机 `192.168.0.10` 上执行：

```text
ROI core tests passed
audio preprocessor tests passed
audio DTX tests passed
bounded audio queue tests passed
transport tests passed
UDP sender tests passed
Snapshot protocol tests passed
Snapshot crop tests passed
rebuild protocol tests passed
rebuild reference selector tests passed
rebuild refresh tests passed
asynchronous RTP sender tests passed
statistics tests passed
```

主机 Python 回归：`python -m unittest tools.test_rebuild_protocol tools.test_live_h265_hud`，`Ran 63 tests ... OK`。相关 Python 文件通过 `py_compile`，离线分析器也通过编译检查。

## 8. 正式 A/B 实测

两组均使用 RK3588 摄像头、`max-frames=480`、真实 H.265、真实 rebuild UDP、音频打开、接收端真实 CUDA Real-ESRGAN x2，单组有效运行约 79.6 秒。由于是连续两次现场采集，不是同一帧/同一动作的配对实验，带宽和渲染统计可比较，但画质指标不是严格 paired gain。

| 指标 | person | adaptive |
|---|---:|---:|
| 有效时长 / sender frames / HUD frames | 79.682 s / 480 / 475 | 79.644 s / 480 / 474 |
| reference 数量 / transfers/s | 118 / 1.481 | 62 / 0.778 |
| reference kind / fallback | FULL / 0 | HEAD / 0 |
| JPEG bytes p50 / p95 / max | 739 / 779.3 / 794 B | 777 / 798 / 800 B |
| JPEG Q p50 / p95 | 40 / 40 | 45 / 45 |
| head width p10 / p50 / p90 | 33 / 35 / 35 px | 58 / 70 / 70 px |
| head height p10 / p50 / p90 | 13 / 14 / 21 px | 34 / 40 / 44 px |
| head area p10 / p50 / p90 | 455 / 490 / 693 px2 | 1977.8 / 2800 / 3010 px2 |
| linear gain p10 / p50 / p90 | 1.000 / 1.000 / 1.000 | 2.082 / 2.450 / 2.512 |
| area gain p10 / p50 / p90 | 1.000 / 1.000 / 1.000 | 4.334 / 6.000 / 6.308 |
| H265 wire p50 / p95 | 22.3 / 27.2 kbps | 19.1 / 27.9 kbps |
| RB wire p50 / p95 | 25.4 / 44.8 kbps | 5.1 / 58.6 kbps |
| V+RB wire p50 / p95 / max | 46.2 / 65.7 / 70.0 kbps | 24.5 / 83.2 / 95.1 kbps |
| audio actual / reserved | 2.560 / 10.2 kbps | 7.920 / 10.2 kbps |
| V+RB+audio+event p50 / p95 / max | 50.428 / 69.928 / 74.228 kbps | 32.718 / 91.418 / 103.318 kbps |
| age / state / content drops | 0.21% / 64.21% / 0.21% | 0.63% / 77.00% / 0% |
| reference delivery p95 | 146.97 ms | 179.22 ms |
| ROI-ESRGAN / ROI-LANCZOS / BASE | 21.474% / 14.105% / 64.421% | 20.886% / 1.055% / 78.059% |
| SR hit / last latency p50 / p95 | 21.474% / 88 / 99 ms | 20.886% / 87 / 99 ms |
| SR stale | 22 | 6 |
| source-to-final PSNR p50 / SSIM p50 | 25.158 dB / 0.8303 | 26.781 dB / 0.8355 |
| render mode transitions | 151 | 21 |

说明：总带宽的 `max` 是 HUD 滚动统计窗口中各项合并后的峰值；adaptive 有一个约 103.3 kbps 的峰值，p95 为 91.418 kbps，未观察到持续超 100 kbps。音频活动和检测数量在两次采集间不同，不能据此归因全部 H265 差异。PSNR/SSIM 是 source-to-final 的匹配区域代理指标，不是独立高质量 ground truth。

## 9. 视觉结果

adaptive 的有效 head area 中位数约 2800 px2，对 person FULL 约 490 px2，面积增益中位数 6.0x，说明“有限字节集中到较小区域”这一机制生效。接收端真实走过 `ROI-ESRGAN`，两组 SR hit 约 21%，SR 延迟 p95 均为 99 ms。

但本次开发板现场画面只有未控动作，代表性 adaptive 样本的 focus 偏向举起的手。由于没有稳定人脸标注和静态/转身/远近/遮挡动作集，本报告不把 proxy PSNR/SSIM 或手部样本当作人脸画质验收结论。

## 10. 100 kbps 验证

发送端仍以 `100000 bps` 为 V+RB+audio 的预算上限，JPEG cap 检查结果为 PASS，正式样本最大 JPEG 为 794 B / 800 B。person 的合并带宽 p50/p95/max 为 50.428/69.928/74.228 kbps；adaptive 为 32.718/91.418/103.318 kbps。adaptive 的单次峰值需要后续把音频/event 峰值纳入更严格的滑动预算审计，但目前 p95 和丢包稳定性没有显示持续越界。

## 11. RK3588 部署

Ubuntu 交叉编译完成：`[100%] Built target rknn_yolov8_seg_cam`。目标二进制 SHA256 为 `990bcc1827d7500289d96dff30f51011430f717eba99fc22472e8be35494baba`，包 SHA256 为 `8b2db6ad69aef6e1b5483ce3b583d7d8a6d05f0943e86ca1820bdb22462d74de`。

已通过校验和传输到 `192.168.0.101`，并以原子目录替换部署到 `/opt/atk/rknn_yolov8_seg_cam`；旧版本保留为 `/opt/atk/rknn_yolov8_seg_cam.backup_goal0901a_20260901_065248`。板端新二进制 SHA 与交叉编译产物一致。smoke test 和两次 480-frame 正式运行均正常退出，未留下运行中的 sender/receiver 进程。

## 12. CUDA Real-ESRGAN

PC 接收端加载 `model/RealESRGAN_x2_dynamic.onnx`，provider 为 `CUDAExecutionProvider`，两次正式运行的 warmup 分别约 1747 ms 和 1706 ms。`output.mp4` 均为约 12 fps、超过 60 秒，视频有效内容为 640x360（文件因默认旋转元数据报告为 360x640）。

## 13. 当前剩余问题

1. 现有 person bbox + segmentation proxy 不能稳定区分脸、手和上半身；本次样本已经暴露手部误选。
2. 两次 A/B 是顺序采集，且 audio、reference 数量、动作和 state drop 不同，不能替代受控 paired test。
3. adaptive 的 reference delivery p95 和 queue delay 高于 person，运行中 state drop 也更高，需要在受控目标存在时继续观察。
4. 仍缺少静态脸、运动脸、转身、近景、远景、遮挡、多人和目标离开/重新进入的实机验收样本。
5. 100 kbps 仍需对音频/event 峰值做严格同一滑动窗口预算约束，避免偶发合并峰值超过上限。

## 14. 默认策略结论

暂不把默认模式切换为 `adaptive`。它已经满足像素密度增益目标（线性中位数约 2.45x、面积中位数约 6.0x），但本次实机证据显示选择器对“脸部”不可靠，且没有完成目标规定的受控低误选率验收。默认继续使用 `person`，可用 `--rebuild-reference-mode=adaptive` 做后续实验。

## 15. 下一阶段建议

先在板端固定同一人、同一背景和同一动作，分别采集静态、运动、转身、近景、远景、遮挡、多人和出入场景；为每个参考样本标注 face/head 是否命中，再重新计算增益、误选率、delivery p95、drop 和 100 kbps 滑动峰值。只有选择器达到目标且视觉结果稳定后再考虑改默认；当前不引入双 patch 或 GFPGAN。

## 16. 工作区与证据

- 正式分析结果：`runs/goal0901a_person2/metrics.json`、`runs/goal0901a_adaptive2/metrics.json`。
- 原始证据：两组 `sender.log`、`hud.log`、`sender_refs`、`refs` 和 `output.mp4` 均保留。
- 已保留用户新增的 `goal0901a.txt`、`模型分析.md`，未修改或回退无关工作区变更。
- 本次未执行 `git reset`、`git checkout`、`git restore`、`git clean`、`git pull` 或 `git push`。
