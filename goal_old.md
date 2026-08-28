在 RK3588 开发板上实现一套基于 YOLOv8-Seg 的 Segmentation-Aware ROI H.265 超低码率实时视频传输系统。
开发板使用Buildroot系统，地址为192.168.0.11，用户名root，密码root。交叉编译环境为ubuntu虚拟机，地址192.169.0.101，用户名alientek，密码1234，你可以自行登录以使用环境。

系统使用现有 YOLOv8-Seg RKNN 推理结果，将目标实例的 segmentation mask 转换为编码重要性区域，并映射到 H.265 编码单元级 ROI/QP 控制，使有限码率优先分配给目标主体和目标边缘，降低背景区域编码质量，在 60 kbps 网络带宽限制下尽可能提升关键目标的视觉质量。

总体数据链路：

Camera / V4L2 / RKISP
→ RGA 缩放与 NV12 图像处理
→ YOLOv8-Seg RKNN 推理
→ Segmentation ROI Mapper
→ 16×16 Importance Map
→ Background / Halo / Core / Edge 分级
→ Relative QP Map / ROI Region
→ RK3588 MPP H.265 Hardware Encoder
→ Rate Pacer / Token Bucket
→ UDP/RTP
→ PC
→ 标准 H.265 解码与显示

核心实现要求：

1. 复用现有 YOLOv8-Seg 板端推理代码，不重新实现模型推理。

2. YOLOv8-Seg 输出至少包含：

   * frame_id
   * timestamp / PTS
   * class_id
   * confidence
   * bbox
   * segmentation mask

3. 正确处理 YOLO 输入 letterbox 坐标转换，将 segmentation mask 从模型坐标系准确映射到实际 H.265 编码分辨率。

4. 实现独立 RoiMapper 模块，将 segmentation mask 转换成 16×16 编码网格 importance map。

5. ROI 至少划分为四级：

   * BACKGROUND：低优先级，建议 relative QP +8
   * HALO：目标外围过渡区，建议 relative QP +2
   * CORE：目标主体，建议 relative QP -4
   * EDGE：目标轮廓，最高优先级，建议 relative QP -8

QP 参数必须可配置，不能硬编码。

6. CORE / EDGE / HALO 根据 segmentation mask 的腐蚀、膨胀、边缘提取以及每个 16×16 Cell 的 mask occupancy 生成，而不是简单将 segmentation mask 转换成 bounding box。

7. 实现 ROI temporal smoothing / hysteresis，减少连续帧 segmentation mask 抖动造成的编码区域 QP 闪烁。

目标出现时允许快速提升编码优先级；目标消失后，应经过若干帧逐步从 EDGE/CORE → HALO → BACKGROUND。

8. 第一阶段使用 Rockchip MPP ROI API 实现：
   Segmentation Mask
   → 16×16 Importance Map
   → ROI Rectangle Merge
   → RoiRegionCfg
   → mpp_enc_roi_add_region()
   → mpp_enc_roi_setup_meta()
   → MPP H.265 Encoder

优先使用 relative QP，且默认 force_intra=0。

9. ROI rectangle 合并必须减少区域数量，并提供 MAX_ROI_REGION 限制。超过限制时按照 EDGE > CORE > HALO > BACKGROUND 的顺序保证重要区域优先保留。

10. H.265 使用 RK3588 MPP 硬件编码，不使用软件 x265 作为正式发送端编码器。

初始实验参数：

* 编码分辨率：320×180
* 编码帧率：3 fps
* 编码格式：H.265/HEVC
* Rate Control：CBR
* H.265 Target Bitrate：约 45 kbps
* GOP：约 30 帧
* 网络物理限制：60 kbps

上述参数全部设计为运行时可配置。

11. 编码线程不得同步等待 YOLO 推理。

采用异步流水线：

Capture Thread
├─→ YOLO/RKNN Thread
│      ↓
│   SegResult
│      ↓
│   RoiManager
│
└─→ Encoder Thread
↓
获取时间上最近且有效的 ROI
↓
MPP Encoder

通过 frame_id 和 PTS 保证 segmentation 结果与编码帧关联。

12. 当当前编码帧没有新的 segmentation 结果时，可以短时间复用历史 ROI Map，但必须设置最大 ROI age，超过阈值后逐渐退化到普通背景编码，不能无限使用过期 mask。

13. 在 MPP 输出之后实现独立 RatePacer。

MPP CBR 负责平均码率控制，RatePacer 使用 Token Bucket 或等效算法限制实际网络发送速率，防止 I 帧/复杂帧形成瞬时 burst 撑爆 60 kbps 限速链路。

14. PC 接收端保持标准化。

PC 不需要 YOLOv8-Seg、不需要 ROI 信息，也不需要特殊解码器。

发送的视频必须保持为标准 H.265 bitstream，使 FFmpeg、GStreamer、libavcodec 等标准 H.265 解码器可以直接解码。

15. 模块至少拆分为：

capture/

* camera_capture

inference/

* yolov8_seg

roi/

* roi_mapper
* roi_temporal
* roi_region_merger

encoder/

* mpp_h265_encoder

transport/

* packetizer
* rate_pacer
* udp_sender

common/

* frame_meta
* config
* statistics

16. RoiMapper 与 YOLO 和 MPP 解耦。

建议核心接口：

RoiMap RoiMapper::build(
const SegResult& segmentation,
int encoder_width,
int encoder_height
);

MPP Encoder 只接受已经处理好的 RoiMap / RoiRegion，不依赖 YOLOv8-Seg 内部数据结构。

17. 所有关键参数通过配置文件或命令行配置，包括：

* encoder width / height
* fps
* target bitrate
* GOP
* qp_min / qp_max
* background_delta_qp
* halo_delta_qp
* core_delta_qp
* edge_delta_qp
* mask occupancy threshold
* erosion radius
* dilation radius
* ROI hold frames
* ROI max age
* MAX_ROI_REGION
* UDP destination
* network pacing bitrate

18. 增加运行时统计和调试日志，至少统计：

* frame_id
* segmentation latency
* segmentation result age
* ROI cell count
* EDGE/CORE/HALO/BACKGROUND 占比
* ROI rectangle count
* H.265 frame type
* encoded frame bytes
* instantaneous bitrate
* average bitrate
* encoder average QP
* send queue size
* end-to-end latency

19. 提供 Debug 模式，可以将最终 16×16 ROI Importance Map 叠加到视频或保存成图片，用于确认 segmentation mask → 编码 ROI 的坐标映射是否正确。

20. 建立三组可重复的对照实验：

A. 普通 H.265
B. YOLO bbox ROI + H.265
C. Segmentation-Aware ROI + H.265

保证三组使用相同输入视频、分辨率、帧率和目标码率。

比较指标至少包括：

* 实际平均码率
* 1 秒峰值码率
* 全图 PSNR / SSIM
* ROI PSNR / SSIM
* Background PSNR / SSIM
* 目标边缘质量
* 平均 QP
* RK3588 CPU / NPU / VPU 占用
* 端到端延迟

最终验收目标：

在网络带宽严格限制为 60 kbps 的环境中，系统能够持续、稳定地从 RK3588 向 PC 传输标准 H.265 视频；相比相同码率下的普通 H.265 编码，Segmentation-Aware ROI 方案应明显提高 YOLOv8-Seg 检测目标主体及轮廓区域的视觉质量，同时允许背景质量主动下降，并且不能产生不可控的码率峰值、ROI 抖动或明显增加端到端延迟。

第一阶段优先完成：

YOLOv8-Seg Mask
→ 坐标映射
→ 16×16 Importance Map
→ CORE / EDGE / HALO / BACKGROUND
→ Temporal Smoothing
→ Rectangle Merge
→ MPP RoiRegionCfg
→ H.265 编码

在第一阶段稳定运行和完成 A/B/C 对比测试之后，再进入第二阶段，研究绕过 Rectangle Merge、直接将 segmentation importance map 映射为 RK3588 VEPU58x H.265 CU/CTU QP Map 的实现。