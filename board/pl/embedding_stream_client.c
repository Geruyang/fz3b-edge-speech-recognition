/* Capture one ordered series of requests on one FZ3B Embedding lease. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <unistd.h>

#include "driver/uapi/fzspeech_embedding_stream.h"

_Static_assert(sizeof(struct fz_embedding_stream_request) == 56, "unexpected stream ABI");
_Static_assert(sizeof(struct fz_embedding_stream_evidence_request) == 48, "unexpected evidence request ABI");
_Static_assert(sizeof(struct fz_embedding_stream_evidence) == FZ_EMBED_STREAM_EVIDENCE_BYTES,
               "unexpected evidence ABI");

static int read_exact_file(const char *name, unsigned char *buffer, size_t size)
{
    FILE *file = fopen(name, "rb");
    if (!file)
        return -1;
    int ok = fread(buffer, 1, size, file) == size && fgetc(file) == EOF && !ferror(file);
    if (fclose(file))
        return -1;
    if (!ok) {
        errno = EINVAL;
        return -1;
    }
    return 0;
}

static int save(int dir, const char *name, const void *buffer, size_t size)
{
    int fd = openat(dir, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0)
        return -1;
    size_t pos = 0;
    while (pos < size) {
        ssize_t n = write(fd, (const unsigned char *)buffer + pos, size - pos);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0) {
            int saved = n < 0 ? errno : EIO;
            close(fd);
            errno = saved;
            return -1;
        }
        pos += (size_t)n;
    }
    return close(fd);
}

static int save_detail(int dev, int dir, int index, const struct fz_embedding_stream_request *job)
{
    struct fz_embedding_stream_evidence *evidence = malloc(sizeof(*evidence));
    if (!evidence)
        return -1;
    struct fz_embedding_stream_evidence_request req = {
        .generation = job->generation, .epoch = job->epoch, .chain_id = job->chain_id,
        .sequence = job->sequence, .evidence_ptr = (uintptr_t)evidence, .evidence_bytes = sizeof(*evidence)
    };
    int rc = ioctl(dev, FZ_EMBED_STREAM_EVIDENCE, &req);
    if (rc == 0) {
        const struct fz_embedding_stream_evidence_header *header = &evidence->header;
        if (header->generation != job->generation || header->epoch != job->epoch ||
            header->chain_id != job->chain_id || header->sequence != job->sequence ||
            header->magic != FZ_EMBED_STREAM_EVIDENCE_MAGIC || header->version != FZ_EMBED_STREAM_EVIDENCE_VERSION ||
            header->evidence_bytes != sizeof(*evidence)) {
            errno = EPROTO;
            rc = -1;
        } else {
            char name[48];
            snprintf(name, sizeof(name), "chunk-%02d.detail", index);
            rc = save(dir, name, evidence, sizeof(*evidence));
        }
    }
    free(evidence);
    return rc;
}

static int capture(int dev, int dir, int count, char **inputs, int details)
{
    uint64_t generation = 0, epoch = 0;
    unsigned char input[FZ_EMBED_STREAM_INPUT_BYTES], output[FZ_EMBED_STREAM_OUTPUT_BYTES];
    unsigned char boot[37], session[37];
    if (read_exact_file("/proc/sys/kernel/random/boot_id", boot, sizeof(boot)) ||
        read_exact_file("/proc/sys/kernel/random/uuid", session, sizeof(session)) ||
        save(dir, "boot.before", boot, sizeof(boot)) || save(dir, "session", session, sizeof(session)))
        return -1;
    for (int index = 0; index < count; ++index) {
        char name[48], receipt[160];
        if (read_exact_file(inputs[index], input, sizeof(input)))
            return -1;
        snprintf(name, sizeof(name), "chunk-%02d.input", index);
        if (save(dir, name, input, sizeof(input)))
            return -1;
        memset(output, 0xa5, sizeof(output));
        struct fz_embedding_stream_request req = {
            .chain_id = (uint64_t)index + 1,
            .input_ptr = (uintptr_t)input, .output_ptr = (uintptr_t)output,
            .input_bytes = sizeof(input), .output_bytes = sizeof(output)
        };
        /* An interrupted ioctl may already have reached hardware. Never repeat it. */
        if (ioctl(dev, FZ_EMBED_STREAM_SUBMIT, &req) < 0)
            return -1;
        if (index == 0) {
            generation = req.generation;
            epoch = req.epoch;
        }
        if (!generation || !epoch || req.generation != generation || req.epoch != epoch ||
            req.chain_id != (uint64_t)index + 1 || req.sequence != (uint32_t)index + 1) {
            errno = EPROTO;
            return -1;
        }
        if (details && save_detail(dev, dir, index, &req))
            return -1;
        snprintf(name, sizeof(name), "chunk-%02d.output", index);
        if (save(dir, name, output, sizeof(output)))
            return -1;
        int length = snprintf(receipt, sizeof(receipt), "%d %" PRIu64 " %" PRIu64 " %" PRIu64 " %u\n",
                              index, generation, epoch, (uint64_t)req.chain_id, req.sequence);
        snprintf(name, sizeof(name), "chunk-%02d.receipt", index);
        if (length < 0 || (size_t)length >= sizeof(receipt) || save(dir, name, receipt, (size_t)length))
            return -1;
    }
    unsigned char after[sizeof(boot)];
    if (read_exact_file("/proc/sys/kernel/random/boot_id", after, sizeof(after)) ||
        memcmp(boot, after, sizeof(boot))) {
        errno = ESTALE;
        return -1;
    }
    return save(dir, "boot.after", after, sizeof(after));
}

int main(int argc, char **argv)
{
    int details = argc > 1 && !strcmp(argv[1], "--details");
    int directory_index = 1 + details;
    if (argc < directory_index + 2) {
        fprintf(stderr, "usage: %s [--details] NEW_EXPORT_DIR INPUT_F32LE...\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *directory = argv[directory_index];
    if (mkdir(directory, 0700)) {
        perror("create fresh export directory");
        return EXIT_FAILURE;
    }
    int dir = open(directory, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    int dev = open(FZ_EMBED_STREAM_DEVICE, O_RDWR | O_CLOEXEC | O_NOFOLLOW);
    struct stat st;
    if (dir < 0 || dev < 0 || fstat(dev, &st) || !S_ISCHR(st.st_mode)) {
        fprintf(stderr, "cannot open the bound Embedding stream device and export directory\n");
        return EXIT_FAILURE;
    }
    int count = argc - directory_index - 1;
    int rc = capture(dev, dir, count, argv + directory_index + 1, details);
    if (rc)
        perror("Embedding stream stopped");
    if (close(dev))
        rc = -1;
    if (!rc && save(dir, "closed", "ok\n", 3))
        rc = -1;
    if (close(dir) || rc)
        return EXIT_FAILURE;
    printf("FZ_EMBED_STREAM_CAPTURE completed=%d\n", count);
    return EXIT_SUCCESS;
}
