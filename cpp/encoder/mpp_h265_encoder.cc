#include "encoder/mpp_h265_encoder.h"

#include <algorithm>
#include <cstring>

#ifdef HAVE_ROCKCHIP_MPP
extern "C" {
#include <rk_mpi.h>
#include <rk_venc_cmd.h>
#include <rk_venc_rc.h>
}
#endif

namespace roi_h265 {
namespace {

int alignTo(int value, int alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

bool isKeyH265AccessUnit(const std::vector<uint8_t> &bytes) {
    for (size_t i = 0; i + 5 < bytes.size(); ++i) {
        size_t code = 0;
        if (bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 1) code = 3;
        else if (bytes[i] == 0 && bytes[i + 1] == 0 && bytes[i + 2] == 0 && bytes[i + 3] == 1) code = 4;
        if (code) {
            const int nal_type = (bytes[i + code] >> 1) & 0x3f;
            if (nal_type >= 16 && nal_type <= 21) return true;
            i += code;
        }
    }
    return false;
}

}  // namespace

#ifdef HAVE_ROCKCHIP_MPP
struct MppEncoderState {
    MppCtx context;
    MppApi *mpi;
    MppEncCfg config;
    MppBufferGroup buffer_group;
    MppBuffer frame_buffer;
    MppBuffer packet_buffer;
    std::vector<MppEncROIRegion> roi_regions;
    MppEncROICfg roi_config;
    int hor_stride;
    int ver_stride;
    size_t frame_bytes;
    size_t packet_bytes;

    MppEncoderState()
        : context(NULL), mpi(NULL), config(NULL), buffer_group(NULL), frame_buffer(NULL),
          packet_buffer(NULL), hor_stride(0), ver_stride(0), frame_bytes(0), packet_bytes(0) {
        std::memset(&roi_config, 0, sizeof(roi_config));
    }
};

namespace {

bool checkMpp(MPP_RET ret, const char *operation, std::string *error) {
    if (ret == MPP_OK) return true;
    if (error) *error = std::string(operation) + " failed: " + std::to_string(ret);
    return false;
}

void destroyMppState(MppEncoderState *state) {
    if (!state) return;
    if (state->packet_buffer) mpp_buffer_put(state->packet_buffer);
    if (state->frame_buffer) mpp_buffer_put(state->frame_buffer);
    if (state->buffer_group) mpp_buffer_group_put(state->buffer_group);
    if (state->config) mpp_enc_cfg_deinit(state->config);
    if (state->context) mpp_destroy(state->context);
    delete state;
}

bool setEncoderConfig(MppEncoderState *state, const EncoderConfig &config, std::string *error) {
    const int width = config.width;
    const int height = config.height;
    const int target = config.target_bitrate_bps;
    const int min_bps = target * 15 / 16;
    const int max_bps = target * 17 / 16;
    const MPP_RET results[] = {
        mpp_enc_cfg_set_s32(state->config, "codec:type", MPP_VIDEO_CodingHEVC),
        mpp_enc_cfg_set_s32(state->config, "prep:width", width),
        mpp_enc_cfg_set_s32(state->config, "prep:height", height),
        mpp_enc_cfg_set_s32(state->config, "prep:hor_stride", state->hor_stride),
        mpp_enc_cfg_set_s32(state->config, "prep:ver_stride", state->ver_stride),
        mpp_enc_cfg_set_s32(state->config, "prep:format", MPP_FMT_YUV420SP),
        mpp_enc_cfg_set_s32(state->config, "rc:mode", MPP_ENC_RC_MODE_CBR),
        mpp_enc_cfg_set_s32(state->config, "rc:fps_in_flex", 0),
        mpp_enc_cfg_set_s32(state->config, "rc:fps_in_num", config.fps),
        // MPP names this historical field "denorm" (not "denom").
        mpp_enc_cfg_set_s32(state->config, "rc:fps_in_denorm", 1),
        mpp_enc_cfg_set_s32(state->config, "rc:fps_out_flex", 0),
        mpp_enc_cfg_set_s32(state->config, "rc:fps_out_num", config.fps),
        mpp_enc_cfg_set_s32(state->config, "rc:fps_out_denorm", 1),
        mpp_enc_cfg_set_s32(state->config, "rc:bps_target", target),
        mpp_enc_cfg_set_s32(state->config, "rc:bps_min", min_bps),
        mpp_enc_cfg_set_s32(state->config, "rc:bps_max", max_bps),
        mpp_enc_cfg_set_s32(state->config, "rc:gop", config.gop),
        mpp_enc_cfg_set_s32(state->config, "rc:max_reenc_times", config.max_reencode_times),
        mpp_enc_cfg_set_s32(state->config, "rc:priority", MPP_ENC_RC_BY_FRM_SIZE_FIRST),
        mpp_enc_cfg_set_s32(state->config, "rc:super_mode", MPP_ENC_RC_SUPER_FRM_REENC),
        mpp_enc_cfg_set_s32(state->config, "rc:super_i_thd", config.super_i_frame_bits),
        mpp_enc_cfg_set_s32(state->config, "rc:super_p_thd", config.super_p_frame_bits),
        mpp_enc_cfg_set_s32(state->config, "rc:qp_init", config.qp_init),
        mpp_enc_cfg_set_s32(state->config, "rc:qp_min", config.qp_min),
        mpp_enc_cfg_set_s32(state->config, "rc:qp_max", config.qp_max),
        mpp_enc_cfg_set_s32(state->config, "rc:qp_min_i", config.qp_min_i),
        mpp_enc_cfg_set_s32(state->config, "rc:qp_max_i", config.qp_max_i),
        mpp_enc_cfg_set_s32(state->config, "rc:refresh_en", config.intra_refresh ? 1 : 0),
        mpp_enc_cfg_set_s32(state->config, "rc:refresh_mode", MPP_ENC_RC_INTRA_REFRESH_ROW),
        mpp_enc_cfg_set_s32(state->config, "rc:refresh_num", config.intra_refresh_rows),
    };
    const char *const keys[] = {
        "codec:type", "prep:width", "prep:height", "prep:hor_stride", "prep:ver_stride",
        "prep:format", "rc:mode", "rc:fps_in_flex", "rc:fps_in_num", "rc:fps_in_denorm",
        "rc:fps_out_flex", "rc:fps_out_num", "rc:fps_out_denorm", "rc:bps_target",
        "rc:bps_min", "rc:bps_max", "rc:gop", "rc:max_reenc_times", "rc:priority",
        "rc:super_mode", "rc:super_i_thd", "rc:super_p_thd", "rc:qp_init", "rc:qp_min",
        "rc:qp_max", "rc:qp_min_i", "rc:qp_max_i", "rc:refresh_en", "rc:refresh_mode",
        "rc:refresh_num",
    };
    for (size_t i = 0; i < sizeof(results) / sizeof(results[0]); ++i) {
        const std::string operation = std::string("mpp_enc_cfg_set ") + keys[i];
        if (!checkMpp(results[i], operation.c_str(), error)) return false;
    }
    if (!checkMpp(state->mpi->control(state->context, MPP_ENC_SET_CFG, state->config),
                  "MPP_ENC_SET_CFG", error)) return false;
    MppEncHeaderMode header_mode = MPP_ENC_HEADER_MODE_EACH_IDR;
    return checkMpp(state->mpi->control(state->context, MPP_ENC_SET_HEADER_MODE, &header_mode),
                    "MPP_ENC_SET_HEADER_MODE", error);
}

}  // namespace
#endif

MppH265Encoder::MppH265Encoder(const EncoderConfig &config, const RoiConfig &roi_config)
    : config_(config), roi_config_(roi_config), state_(NULL) {}

MppH265Encoder::~MppH265Encoder() { shutdown(); }

bool MppH265Encoder::available() const {
#ifdef HAVE_ROCKCHIP_MPP
    return true;
#else
    return false;
#endif
}

bool MppH265Encoder::initialize(std::string *error) {
    shutdown();
#ifndef HAVE_ROCKCHIP_MPP
    if (error) *error = "Rockchip MPP headers/library were not found at build time";
    return false;
#else
    MppEncoderState *state = new MppEncoderState;
    state->hor_stride = alignTo(config_.width, 64);
    state->ver_stride = alignTo(config_.height, 64);
    state->frame_bytes = static_cast<size_t>(state->hor_stride) * state->ver_stride * 3 / 2;
    // A compressed IDR can be larger than the NV12 frame for unusual runtime
    // inputs. Keep a bounded minimum buffer rather than truncating MPP output.
    state->packet_bytes = std::max<size_t>(state->frame_bytes, 1024U * 1024U);
    if (!checkMpp(mpp_create(&state->context, &state->mpi), "mpp_create", error) ||
        !checkMpp(mpp_init(state->context, MPP_CTX_ENC, MPP_VIDEO_CodingHEVC), "mpp_init HEVC", error) ||
        !checkMpp(mpp_enc_cfg_init(&state->config), "mpp_enc_cfg_init", error) ||
        !checkMpp(state->mpi->control(state->context, MPP_ENC_GET_CFG, state->config),
                  "MPP_ENC_GET_CFG", error) ||
        !setEncoderConfig(state, config_, error) ||
        !checkMpp(mpp_buffer_group_get_internal(&state->buffer_group, MPP_BUFFER_TYPE_DRM),
                  "mpp_buffer_group_get_internal", error) ||
        !checkMpp(mpp_buffer_get(state->buffer_group, &state->frame_buffer, state->frame_bytes),
                  "mpp_buffer_get frame", error) ||
        !checkMpp(mpp_buffer_get(state->buffer_group, &state->packet_buffer, state->packet_bytes),
                  "mpp_buffer_get packet", error)) {
        destroyMppState(state);
        return false;
    }
    state_ = state;
    return true;
#endif
}

bool MppH265Encoder::encode(const FramePacket &frame, const std::vector<RoiRegion> &regions,
                            EncodedAccessUnit *output, std::string *error) {
#ifndef HAVE_ROCKCHIP_MPP
    (void)frame; (void)regions; (void)output;
    if (error) *error = "MPP encoder is unavailable in this build";
    return false;
#else
    MppEncoderState *state = static_cast<MppEncoderState *>(state_);
    if (!state || !output || frame.encoder_width != config_.width || frame.encoder_height != config_.height ||
        frame.nv12.size() != static_cast<size_t>(config_.width) * config_.height * 3 / 2) {
        if (error) *error = "invalid MPP frame or encoder state";
        return false;
    }
    uint8_t *destination = static_cast<uint8_t *>(mpp_buffer_get_ptr(state->frame_buffer));
    if (!destination) {
        if (error) *error = "MPP input buffer has no CPU mapping";
        return false;
    }
    // The board's MPP runtime exposes CPU-mapped DRM buffers but predates the
    // mpp_buffer_sync_{begin,end}_f helpers declared by the newer SDK headers.
    // Its encoder examples write through mpp_buffer_get_ptr directly; keep the
    // compatible path here instead of creating an unresolved runtime symbol.
    std::memset(destination, 0, state->frame_bytes);
    for (int y = 0; y < config_.height; ++y) {
        std::memcpy(destination + static_cast<size_t>(y) * state->hor_stride,
                    &frame.nv12[static_cast<size_t>(y) * config_.width], config_.width);
    }
    const size_t dst_uv_offset = static_cast<size_t>(state->hor_stride) * state->ver_stride;
    const size_t src_uv_offset = static_cast<size_t>(config_.width) * config_.height;
    for (int y = 0; y < config_.height / 2; ++y) {
        std::memcpy(destination + dst_uv_offset + static_cast<size_t>(y) * state->hor_stride,
                    &frame.nv12[src_uv_offset + static_cast<size_t>(y) * config_.width], config_.width);
    }

    MppFrame mpp_frame = NULL;
    MppPacket packet = NULL;
    if (!checkMpp(mpp_frame_init(&mpp_frame), "mpp_frame_init", error)) return false;
    mpp_frame_set_width(mpp_frame, config_.width);
    mpp_frame_set_height(mpp_frame, config_.height);
    mpp_frame_set_hor_stride(mpp_frame, state->hor_stride);
    mpp_frame_set_ver_stride(mpp_frame, state->ver_stride);
    mpp_frame_set_fmt(mpp_frame, MPP_FMT_YUV420SP);
    mpp_frame_set_pts(mpp_frame, frame.meta.pts_us);
    mpp_frame_set_buffer(mpp_frame, state->frame_buffer);
    MppMeta meta = mpp_frame_get_meta(mpp_frame);
    if (!checkMpp(mpp_packet_init_with_buffer(&packet, state->packet_buffer), "mpp_packet_init", error)) {
        mpp_frame_deinit(&mpp_frame);
        return false;
    }
    mpp_packet_set_length(packet, 0);
    mpp_meta_set_packet(meta, KEY_OUTPUT_PACKET, packet);
    if (regions.size() > static_cast<size_t>(std::min(roi_config_.max_regions, H265E_MAX_ROI_NUMBER))) {
        if (error) *error = "ROI rectangle count exceeds the configured or MPP H.265 limit";
        mpp_packet_deinit(&packet);
        mpp_frame_deinit(&mpp_frame);
        return false;
    }
    // MPP 1.3.9 exposes ROI through KEY_ROI_DATA but does not install the
    // newer mpp_enc_roi_utils helper. Keep this storage in encoder state as
    // required by MppEncROIRegion: MPP may retain the pointer while encoding.
    state->roi_regions.clear();
    state->roi_regions.reserve(regions.size());
    for (size_t i = 0; i < regions.size(); ++i) {
        MppEncROIRegion region;
        std::memset(&region, 0, sizeof(region));
        region.x = static_cast<RK_U16>(regions[i].x);
        region.y = static_cast<RK_U16>(regions[i].y);
        region.w = static_cast<RK_U16>(regions[i].width);
        region.h = static_cast<RK_U16>(regions[i].height);
        region.intra = regions[i].force_intra ? 1 : 0;
        region.quality = static_cast<RK_S16>(regions[i].delta_qp);
        region.abs_qp_en = 0;  // Relative QP is required by the low-bitrate profile.
        state->roi_regions.push_back(region);
    }
    state->roi_config.number = static_cast<RK_U32>(state->roi_regions.size());
    state->roi_config.regions = state->roi_regions.empty() ? NULL : &state->roi_regions[0];
    if (!checkMpp(mpp_meta_set_ptr(meta, KEY_ROI_DATA, &state->roi_config),
                  "mpp_meta_set_ptr KEY_ROI_DATA", error) ||
        !checkMpp(state->mpi->encode_put_frame(state->context, mpp_frame), "encode_put_frame", error)) {
        mpp_packet_deinit(&packet);
        mpp_frame_deinit(&mpp_frame);
        return false;
    }
    mpp_frame_deinit(&mpp_frame);
    if (!checkMpp(state->mpi->encode_get_packet(state->context, &packet), "encode_get_packet", error)) {
        mpp_packet_deinit(&packet);
        return false;
    }
    const uint8_t *payload = static_cast<const uint8_t *>(mpp_packet_get_pos(packet));
    const size_t length = mpp_packet_get_length(packet);
    if (!payload || length == 0) {
        if (error) *error = "MPP returned an empty H.265 access unit";
        mpp_packet_deinit(&packet);
        return false;
    }
    output->frame = frame.meta;
    output->bytes.assign(payload, payload + length);
    output->key_frame = isKeyH265AccessUnit(output->bytes);
    output->average_qp = -1;
    output->realtime_bitrate_bps = 0;
    if (mpp_packet_has_meta(packet)) {
        MppMeta packet_meta = mpp_packet_get_meta(packet);
        mpp_meta_get_s32(packet_meta, KEY_ENC_AVERAGE_QP, &output->average_qp);
        // MPP 1.3.9 does not expose the newer KEY_ENC_BPS_RT meta key.
        // RuntimeStatistics computes instantaneous and average bitrate from
        // actual encoded bytes when this SDK-provided metric is unavailable.
    }
    mpp_packet_deinit(&packet);
    return true;
#endif
}

bool MppH265Encoder::requestIdr(std::string *error) {
#ifndef HAVE_ROCKCHIP_MPP
    if (error) *error = "MPP encoder is unavailable in this build";
    return false;
#else
    MppEncoderState *state = static_cast<MppEncoderState *>(state_);
    if (!state) {
        if (error) *error = "MPP encoder is not initialized";
        return false;
    }
    return checkMpp(state->mpi->control(state->context, MPP_ENC_SET_IDR_FRAME, NULL),
                    "MPP_ENC_SET_IDR_FRAME", error);
#endif
}

void MppH265Encoder::shutdown() {
#ifdef HAVE_ROCKCHIP_MPP
    destroyMppState(static_cast<MppEncoderState *>(state_));
#endif
    state_ = NULL;
}

}  // namespace roi_h265
