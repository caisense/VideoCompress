#include "audio/arecord_capture.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <spawn.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace roi_h265 {

ArecordCapture::ArecordCapture() : read_fd_(-1), child_pid_(-1), channels_(0) {}

ArecordCapture::~ArecordCapture() { close(); }

bool ArecordCapture::open(const std::string &device, int sample_rate_hz, int channels,
                          std::string *error) {
    close();
    if (device.empty() || sample_rate_hz <= 0 || channels <= 0) {
        if (error) *error = "invalid arecord device, sample rate, or channel count";
        return false;
    }
    int pcm_pipe[2] = {-1, -1};
    if (::pipe(pcm_pipe) != 0) {
        if (pcm_pipe[0] >= 0) ::close(pcm_pipe[0]);
        if (pcm_pipe[1] >= 0) ::close(pcm_pipe[1]);
        if (error) *error = std::string("cannot create arecord pipes: ") + std::strerror(errno);
        return false;
    }

    // This sender already has a video worker thread by the time microphone
    // capture is enabled.  posix_spawnp avoids the unsafe fork-then-exec path
    // in a multithreaded process, while preserving the board manual's arecord
    // device/rate/format/channel selection exactly.
    posix_spawn_file_actions_t actions;
    int spawn_result = ::posix_spawn_file_actions_init(&actions);
    const bool actions_initialized = spawn_result == 0;
    if (spawn_result == 0) {
        spawn_result = ::posix_spawn_file_actions_addclose(&actions, pcm_pipe[0]);
    }
    if (spawn_result == 0) {
        spawn_result = ::posix_spawn_file_actions_adddup2(&actions, pcm_pipe[1], STDOUT_FILENO);
    }
    if (spawn_result == 0) {
        spawn_result = ::posix_spawn_file_actions_addclose(&actions, pcm_pipe[1]);
    }
    if (spawn_result != 0) {
        ::close(pcm_pipe[0]);
        ::close(pcm_pipe[1]);
        if (actions_initialized) (void)::posix_spawn_file_actions_destroy(&actions);
        if (error) *error = std::string("cannot configure arecord startup: ") +
            std::strerror(spawn_result);
        return false;
    }

    char rate_text[16];
    char channels_text[16];
    std::snprintf(rate_text, sizeof(rate_text), "%d", sample_rate_hz);
    std::snprintf(channels_text, sizeof(channels_text), "%d", channels);
    char *const argv[] = {
        const_cast<char *>("arecord"), const_cast<char *>("-D"),
        const_cast<char *>(device.c_str()), const_cast<char *>("-t"),
        const_cast<char *>("raw"), const_cast<char *>("-r"), rate_text,
        const_cast<char *>("-f"), const_cast<char *>("S16_LE"),
        const_cast<char *>("-c"), channels_text, NULL};
    pid_t pid = -1;
    spawn_result = ::posix_spawnp(&pid, "arecord", &actions, NULL, argv, environ);
    (void)::posix_spawn_file_actions_destroy(&actions);
    if (spawn_result != 0) {
        ::close(pcm_pipe[0]);
        ::close(pcm_pipe[1]);
        if (error) *error = std::string("cannot execute arecord: ") + std::strerror(spawn_result);
        return false;
    }
    ::close(pcm_pipe[1]);
    read_fd_ = pcm_pipe[0];
    child_pid_ = pid;
    channels_ = channels;
    return true;
}

bool ArecordCapture::readInterleaved(int16_t *samples, size_t frames, std::string *error) {
    if (read_fd_ < 0 || !samples || frames == 0 || channels_ <= 0) {
        if (error) *error = "arecord capture is not open";
        return false;
    }
    const size_t expected = frames * static_cast<size_t>(channels_) * sizeof(int16_t);
    uint8_t *destination = reinterpret_cast<uint8_t *>(samples);
    size_t offset = 0;
    while (offset < expected) {
        const ssize_t count = ::read(read_fd_, destination + offset, expected - offset);
        if (count > 0) {
            offset += static_cast<size_t>(count);
            continue;
        }
        if (count == 0) {
            if (error) *error = "arecord ended before supplying PCM";
            return false;
        }
        if (errno == EINTR) continue;
        if (error) *error = std::string("arecord PCM read failed: ") + std::strerror(errno);
        return false;
    }
    return true;
}

void ArecordCapture::requestStop() {
    if (child_pid_ > 0) (void)::kill(child_pid_, SIGTERM);
}

void ArecordCapture::close() {
    requestStop();
    if (read_fd_ >= 0) {
        ::close(read_fd_);
        read_fd_ = -1;
    }
    if (child_pid_ > 0) {
        int status = 0;
        while (::waitpid(child_pid_, &status, 0) < 0 && errno == EINTR) {}
        child_pid_ = -1;
    }
    channels_ = 0;
}

}  // namespace roi_h265
