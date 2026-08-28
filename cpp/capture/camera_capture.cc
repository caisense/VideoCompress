#include "capture/camera_capture.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <linux/videodev2.h>
#include <vector>

#include <opencv2/opencv.hpp>

#include "image_utils.h"

namespace roi_h265 {

namespace {

struct V4l2MappedBuffer {
    void *address;
    size_t length;

    V4l2MappedBuffer() : address(MAP_FAILED), length(0) {}
};

int ioctlRetry(int fd, unsigned long request, void *argument) {
    int result;
    do {
        result = ioctl(fd, request, argument);
    } while (result < 0 && errno == EINTR);
    return result;
}

std::string errnoMessage(const std::string &operation) {
    return operation + ": " + std::strerror(errno);
}

}  // namespace

struct CameraCapture::Impl {
    cv::VideoCapture device;
    uint64_t next_frame_id;
    int v4l2_fd;
    bool v4l2_streaming;
    int v4l2_width;
    int v4l2_height;
    int v4l2_stride;
    std::vector<V4l2MappedBuffer> v4l2_buffers;

    Impl()
        : next_frame_id(0), v4l2_fd(-1), v4l2_streaming(false),
          v4l2_width(0), v4l2_height(0), v4l2_stride(0) {}

    bool queueV4l2Buffer(unsigned int index, std::string *error) {
        if (index >= v4l2_buffers.size()) {
            if (error) *error = "V4L2 returned an out-of-range buffer index";
            return false;
        }
        v4l2_buffer buffer;
        v4l2_plane plane;
        std::memset(&buffer, 0, sizeof(buffer));
        std::memset(&plane, 0, sizeof(plane));
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        buffer.length = 1;
        buffer.m.planes = &plane;
        plane.length = v4l2_buffers[index].length;
        if (ioctlRetry(v4l2_fd, VIDIOC_QBUF, &buffer) < 0) {
            if (error) *error = errnoMessage("VIDIOC_QBUF");
            return false;
        }
        return true;
    }

    void closeV4l2() {
        if (v4l2_fd < 0) return;
        if (v4l2_streaming) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            ioctlRetry(v4l2_fd, VIDIOC_STREAMOFF, &type);
        }
        for (size_t i = 0; i < v4l2_buffers.size(); ++i) {
            if (v4l2_buffers[i].address != MAP_FAILED)
                munmap(v4l2_buffers[i].address, v4l2_buffers[i].length);
        }
        v4l2_buffers.clear();
        ::close(v4l2_fd);
        v4l2_fd = -1;
        v4l2_streaming = false;
        v4l2_width = 0;
        v4l2_height = 0;
        v4l2_stride = 0;
    }

    bool openV4l2(const std::string &path, int requested_width, int requested_height,
                  std::string *error) {
        closeV4l2();
        v4l2_fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
        if (v4l2_fd < 0) {
            if (error) *error = errnoMessage("open " + path);
            return false;
        }
        v4l2_capability capability;
        std::memset(&capability, 0, sizeof(capability));
        if (ioctlRetry(v4l2_fd, VIDIOC_QUERYCAP, &capability) < 0 ||
            !(capability.device_caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) ||
            !(capability.device_caps & V4L2_CAP_STREAMING)) {
            if (error) *error = "not a streaming multi-planar V4L2 capture node: " + path;
            closeV4l2();
            return false;
        }
        v4l2_format format;
        std::memset(&format, 0, sizeof(format));
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        format.fmt.pix_mp.width = requested_width;
        format.fmt.pix_mp.height = requested_height;
        format.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
        format.fmt.pix_mp.field = V4L2_FIELD_NONE;
        format.fmt.pix_mp.num_planes = 1;
        if (ioctlRetry(v4l2_fd, VIDIOC_S_FMT, &format) < 0 ||
            format.fmt.pix_mp.pixelformat != V4L2_PIX_FMT_NV12 ||
            format.fmt.pix_mp.num_planes != 1) {
            if (error) *error = "V4L2 node does not accept single-plane NV12: " + path;
            closeV4l2();
            return false;
        }
        v4l2_width = static_cast<int>(format.fmt.pix_mp.width);
        v4l2_height = static_cast<int>(format.fmt.pix_mp.height);
        v4l2_stride = static_cast<int>(format.fmt.pix_mp.plane_fmt[0].bytesperline);
        if (v4l2_width <= 0 || v4l2_height <= 0 || v4l2_stride < v4l2_width) {
            if (error) *error = "invalid NV12 layout returned by " + path;
            closeV4l2();
            return false;
        }
        v4l2_requestbuffers request;
        std::memset(&request, 0, sizeof(request));
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        request.memory = V4L2_MEMORY_MMAP;
        request.count = 3;
        if (ioctlRetry(v4l2_fd, VIDIOC_REQBUFS, &request) < 0 || request.count < 2) {
            if (error) *error = errnoMessage("VIDIOC_REQBUFS");
            closeV4l2();
            return false;
        }
        v4l2_buffers.resize(request.count);
        for (unsigned int i = 0; i < request.count; ++i) {
            v4l2_buffer buffer;
            v4l2_plane plane;
            std::memset(&buffer, 0, sizeof(buffer));
            std::memset(&plane, 0, sizeof(plane));
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            buffer.memory = V4L2_MEMORY_MMAP;
            buffer.index = i;
            buffer.length = 1;
            buffer.m.planes = &plane;
            if (ioctlRetry(v4l2_fd, VIDIOC_QUERYBUF, &buffer) < 0) {
                if (error) *error = errnoMessage("VIDIOC_QUERYBUF");
                closeV4l2();
                return false;
            }
            v4l2_buffers[i].length = plane.length;
            v4l2_buffers[i].address = mmap(NULL, plane.length, PROT_READ | PROT_WRITE,
                                           MAP_SHARED, v4l2_fd, plane.m.mem_offset);
            if (v4l2_buffers[i].address == MAP_FAILED) {
                if (error) *error = errnoMessage("mmap V4L2 buffer");
                closeV4l2();
                return false;
            }
            if (!queueV4l2Buffer(i, error)) {
                closeV4l2();
                return false;
            }
        }
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        if (ioctlRetry(v4l2_fd, VIDIOC_STREAMON, &type) < 0) {
            if (error) *error = errnoMessage("VIDIOC_STREAMON");
            closeV4l2();
            return false;
        }
        v4l2_streaming = true;
        return true;
    }

    bool readV4l2(cv::Mat *bgr, std::string *error) {
        pollfd descriptor;
        std::memset(&descriptor, 0, sizeof(descriptor));
        descriptor.fd = v4l2_fd;
        descriptor.events = POLLIN;
        const int poll_result = poll(&descriptor, 1, 2000);
        if (poll_result <= 0) {
            if (error) *error = poll_result == 0 ? "V4L2 frame acquisition timed out" : errnoMessage("poll V4L2");
            return false;
        }
        v4l2_buffer buffer;
        v4l2_plane plane;
        std::memset(&buffer, 0, sizeof(buffer));
        std::memset(&plane, 0, sizeof(plane));
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.length = 1;
        buffer.m.planes = &plane;
        if (ioctlRetry(v4l2_fd, VIDIOC_DQBUF, &buffer) < 0) {
            if (error) *error = errnoMessage("VIDIOC_DQBUF");
            return false;
        }
        if (buffer.index >= v4l2_buffers.size() ||
            plane.bytesused < static_cast<unsigned int>(v4l2_stride * v4l2_height * 3 / 2)) {
            if (error) *error = "V4L2 returned an incomplete NV12 frame";
            queueV4l2Buffer(buffer.index, NULL);
            return false;
        }
        const uint8_t *source = static_cast<const uint8_t *>(v4l2_buffers[buffer.index].address);
        cv::Mat tightly_packed(v4l2_height * 3 / 2, v4l2_width, CV_8UC1);
        for (int row = 0; row < v4l2_height; ++row) {
            std::memcpy(tightly_packed.data + static_cast<size_t>(row) * v4l2_width,
                        source + static_cast<size_t>(row) * v4l2_stride, v4l2_width);
        }
        const size_t source_uv = static_cast<size_t>(v4l2_stride) * v4l2_height;
        const size_t destination_uv = static_cast<size_t>(v4l2_width) * v4l2_height;
        for (int row = 0; row < v4l2_height / 2; ++row) {
            std::memcpy(tightly_packed.data + destination_uv + static_cast<size_t>(row) * v4l2_width,
                        source + source_uv + static_cast<size_t>(row) * v4l2_stride, v4l2_width);
        }
        if (!queueV4l2Buffer(buffer.index, error)) return false;
        cv::cvtColor(tightly_packed, *bgr, cv::COLOR_YUV2BGR_NV12);
        return !bgr->empty();
    }
};

namespace {

uint64_t nowMicros() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool numericDevice(const std::string &device, int *index) {
    if (device.empty()) return false;
    char *end = NULL;
    const long parsed = std::strtol(device.c_str(), &end, 10);
    if (!end || *end != '\0' || parsed < 0) return false;
    *index = static_cast<int>(parsed);
    return true;
}

}  // namespace

CameraCapture::CameraCapture(const CameraConfig &camera, const EncoderConfig &encoder)
    : camera_(camera), encoder_(encoder), impl_(NULL) {}

CameraCapture::~CameraCapture() { close(); }

bool CameraCapture::open(std::string *error) {
    close();
    impl_ = new Impl;
    int index = 0;
    const std::string v4l2_path = numericDevice(camera_.device, &index)
        ? "/dev/video" + std::to_string(index) : camera_.device;
    // RKISP mainpath nodes are V4L2 multi-planar.  OpenCV 4.5.4's V4L2
    // backend rejects them even though they stream correctly, so try the
    // native NV12 capture path first.  Files remain on the OpenCV path.
    if (camera_.input_video.empty() &&
        impl_->openV4l2(v4l2_path, camera_.width, camera_.height, error)) {
        return true;
    }
    const bool opened = !camera_.input_video.empty()
        ? impl_->device.open(camera_.input_video)
        : (numericDevice(camera_.device, &index)
            ? impl_->device.open(index, cv::CAP_V4L2)
            : impl_->device.open(camera_.device, cv::CAP_V4L2));
    if (!opened) {
        if (error) *error = "cannot open " + (camera_.input_video.empty() ? camera_.device : camera_.input_video);
        close();
        return false;
    }
    impl_->device.set(cv::CAP_PROP_FRAME_WIDTH, camera_.width);
    impl_->device.set(cv::CAP_PROP_FRAME_HEIGHT, camera_.height);
    EncoderConfig encoder;
    {
        std::lock_guard<std::mutex> lock(encoder_mutex_);
        encoder = encoder_;
    }
    impl_->device.set(cv::CAP_PROP_FPS, encoder.fps);
    return true;
}

bool CameraCapture::read(FramePacket *frame, std::string *error) {
    if (!impl_ || !frame) {
        if (error) *error = "camera is not open";
        return false;
    }
    EncoderConfig encoder;
    {
        std::lock_guard<std::mutex> lock(encoder_mutex_);
        encoder = encoder_;
    }
    cv::Mat bgr;
    const bool read_ok = impl_->v4l2_fd >= 0
        ? impl_->readV4l2(&bgr, error)
        : impl_->device.read(bgr);
    if (!read_ok || bgr.empty()) {
        if (error) *error = "V4L2 frame acquisition failed";
        return false;
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (!rgb.isContinuous()) rgb = rgb.clone();

    FramePacket converted;
    converted.meta.frame_id = impl_->next_frame_id++;
    converted.meta.pts_us = nowMicros();
    converted.meta.capture_time_us = converted.meta.pts_us;
    converted.source_width = rgb.cols;
    converted.source_height = rgb.rows;
    converted.encoder_width = encoder.width;
    converted.encoder_height = encoder.height;
    converted.rgb.assign(rgb.data, rgb.data + static_cast<size_t>(rgb.cols) * rgb.rows * 3);
    converted.nv12.assign(static_cast<size_t>(encoder.width) * encoder.height * 3 / 2, 0);

    image_buffer_t source;
    std::memset(&source, 0, sizeof(source));
    source.width = converted.source_width;
    source.height = converted.source_height;
    source.width_stride = converted.source_width;
    source.height_stride = converted.source_height;
    source.format = IMAGE_FORMAT_RGB888;
    source.virt_addr = converted.rgb.data();
    source.size = static_cast<int>(converted.rgb.size());
    source.fd = -1;
    image_buffer_t destination;
    std::memset(&destination, 0, sizeof(destination));
    destination.width = encoder.width;
    destination.height = encoder.height;
    destination.width_stride = encoder.width;
    destination.height_stride = encoder.height;
    destination.format = IMAGE_FORMAT_YUV420SP_NV12;
    destination.virt_addr = converted.nv12.data();
    destination.size = static_cast<int>(converted.nv12.size());
    destination.fd = -1;
    image_rect_t source_rect;
    source_rect.left = 0;
    source_rect.top = 0;
    source_rect.right = source.width - 1;
    source_rect.bottom = source.height - 1;
    image_rect_t destination_rect;
    destination_rect.left = 0;
    destination_rect.top = 0;
    destination_rect.right = destination.width - 1;
    destination_rect.bottom = destination.height - 1;
    if (convert_image(&source, &destination, &source_rect, &destination_rect, 0) != 0) {
        if (error) *error = "RGA RGB-to-NV12 scale failed";
        return false;
    }
    // Keep the board preview and RKNN input in color, but make the encoded
    // stream effectively monochrome for the 60 kbps visual profile. NV12 UV
    // is interleaved; neutral chroma preserves luma detail and compresses
    // color variation to nearly zero without changing the PC decoder.
    if (encoder.grayscale_encode) {
        const size_t luma_bytes = static_cast<size_t>(encoder.width) * encoder.height;
        std::fill(converted.nv12.begin() + luma_bytes, converted.nv12.end(), 128);
    }
    *frame = converted;
    return true;
}

void CameraCapture::updateEncoderConfig(const EncoderConfig &encoder) {
    std::lock_guard<std::mutex> lock(encoder_mutex_);
    encoder_ = encoder;
}

void CameraCapture::close() {
    if (!impl_) return;
    impl_->closeV4l2();
    if (impl_->device.isOpened()) impl_->device.release();
    delete impl_;
    impl_ = NULL;
}

}  // namespace roi_h265
