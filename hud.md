HUD 不读取板端日志；它直接解析 PC 收到的 RTP 包，并监视本地 FFmpeg 解码结果。核心实现位于 [live_h265_hud.py](D:/workspace/atk_yolov8_seg_cam_v4/tools/live_h265_hud.py)。

| HUD 项                    | 获取来源                                                     |
| ------------------------- | ------------------------------------------------------------ |
| RX fps                    | RTP 的 Marker 位：一帧 H.265 Access Unit 结束记一次，统计最近 1 秒 |
| Decode fps                | HUD 送入 FFmpeg 后，收到一张已解码 BMP 图像记一次            |
| RTP kbps                  | 最近 1 秒收到的原始 RTP/UDP 载荷字节数                       |
| Wire kbps                 | RTP 字节数再估算 IPv4、UDP、以太网帧头、FCS、前导码和帧间隙后的线速 |
| P / I fps、P/I total      | 解析 H.265 RTP 负载中的 NAL 类型；结合 Marker 位按完整帧计数 |
| Packets / Lost / Reorder  | RTP 固定头中的 16 位 sequence number 连续性                  |
| Decode errors             | 监视 FFmpeg stderr 中的参考帧、NAL、解码错误文本             |
| Last IDR                  | 检测到 IDR/IRAP NAL 类型后记录本地时间                       |
| Profile、分辨率、FPS、Gen | 每个 RTP 包的私有 `RO` 扩展（`0x524F`），其中携带档位、宽高、帧率和配置代次 |
| Display age               | PC 本地“最后一帧解码完成”到当前显示的时间，不是严格端到端时延 |

所以 HUD 不只是叠字：它会解析 RTP 固定头、RTP 扩展、H.265 NAL 和 FFmpeg 解码状态。`Display age` 只能反映接收端新鲜度；要严格测量板端采集到 PC 显示的端到端延迟，还需要在线路中携带并对齐发送端时间戳。