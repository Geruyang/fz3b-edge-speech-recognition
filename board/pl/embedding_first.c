/* First FZASR015 Embedding diagnostic. Running with just FIXTURE is offline. */
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <fcntl.h>
#include <linux/magic.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <time.h>
#include <unistd.h>

#include "driver/fzspeech_job.h"
#endif

struct asset {
    const char *name;
    size_t bytes;
    const char *sha256;
};

/* Frozen by reports/t015/embedding-job/fixture.json. */
static const struct asset assets[] = {
    {"weights.bin", 688128, "544f3a212863c84f61b57da22ecbfcabdc730f4b0499415cf95cc781c67bb108"},
    {"weight_scales.f32le", 1696, "262d579d1988395a3139e9eba6e01b53c99862aa0c629bc916cb1e7166433b7b"},
    {"bias.f32le", 1696, "cb98489956366cc63c3a7479b306f9157dfe1281818f9af4786b1aa98f425c1b"},
    {"input.f32le", 12480, "9445583357bbc09b7bc9c1e1ccfffa7fd799fbdd9691d9dd5099996fbd70e82a"},
    {"frozen_w8.f32le", 16384, "cdb16b75c724115106dfee2409e930672780710be860c85783d00b19e15e6c25"},
    {"independent_lut.f32le", 16384, "136f6fd82dbcec5160e2956bcbe05bc3a2bfc4cdb414ae597a21d1c92992fe87"},
};

static uint32_t rotate_right(uint32_t value, unsigned shift)
{
    return (value >> shift) | (value << (32 - shift));
}

/* SHA-256 compression (FIPS 180-4). Keep the static client independent of the
 * target's OpenSSL and of any external command on its root filesystem. */
static void sha256_block(uint32_t hash[8], const unsigned char block[64])
{
    static const uint32_t constant[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };
    uint32_t word[64];
    uint32_t a = hash[0], b = hash[1], c = hash[2], d = hash[3];
    uint32_t e = hash[4], f = hash[5], g = hash[6], h = hash[7];

    for (unsigned i = 0; i < 16; ++i) {
        const unsigned char *p = block + 4 * i;
        word[i] = (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3];
    }
    for (unsigned i = 16; i < 64; ++i) {
        uint32_t x = word[i - 15], y = word[i - 2];
        uint32_t s0 = rotate_right(x, 7) ^ rotate_right(x, 18) ^ (x >> 3);
        uint32_t s1 = rotate_right(y, 17) ^ rotate_right(y, 19) ^ (y >> 10);
        word[i] = word[i - 16] + s0 + word[i - 7] + s1;
    }
    for (unsigned i = 0; i < 64; ++i) {
        uint32_t upper_e = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        uint32_t upper_a = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        uint32_t choice = (e & f) ^ (~e & g);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t1 = h + upper_e + choice + constant[i] + word[i];
        uint32_t t2 = upper_a + majority;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    const uint32_t state[] = {a, b, c, d, e, f, g, h};
    for (unsigned i = 0; i < 8; ++i)
        hash[i] += state[i];
}

static int verify_asset(const char *directory, const struct asset *asset)
{
    char path[4096], hex[65];
    unsigned char block[64];
    uint32_t hash[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    int length = snprintf(path, sizeof(path), "%s/%s", directory, asset->name);
    if (length < 0 || (size_t)length >= sizeof(path)) {
        fprintf(stderr, "%s: fixture path too long\n", asset->name);
        return -1;
    }
    FILE *file = fopen(path, "rb");
    if (!file) {
        perror(path);
        return -1;
    }
    size_t remaining = asset->bytes;
    while (remaining >= sizeof(block)) {
        if (fread(block, 1, sizeof(block), file) != sizeof(block))
            break;
        sha256_block(hash, block);
        remaining -= sizeof(block);
    }
    int correct_length = remaining < sizeof(block) &&
                         fread(block, 1, remaining, file) == remaining &&
                         fgetc(file) == EOF && !ferror(file);
    fclose(file);
    if (!correct_length) {
        fprintf(stderr, "%s: invalid length or read error\n", asset->name);
        return -1;
    }
    memset(block + remaining, 0, sizeof(block) - remaining);
    block[remaining] = 0x80;
    if (remaining >= 56) {
        sha256_block(hash, block);
        memset(block, 0, sizeof(block));
    }
    uint64_t bits = (uint64_t)asset->bytes * 8;
    for (unsigned i = 0; i < 8; ++i)
        block[63 - i] = (unsigned char)(bits >> (8 * i));
    sha256_block(hash, block);
    for (unsigned i = 0; i < 8; ++i)
        snprintf(hex + 8 * i, 9, "%08x", hash[i]);
    if (strcmp(hex, asset->sha256)) {
        fprintf(stderr, "%s: SHA-256 mismatch\n", asset->name);
        return -1;
    }
    return 0;
}

#ifdef __linux__
static int read_node(int dir, const char *name, void *data, size_t max_bytes)
{
    int fd = openat(dir, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0)
        return -1;
    size_t used = 0;
    while (used < max_bytes) {
        ssize_t bytes = read(fd, (unsigned char *)data + used, max_bytes - used);
        if (bytes == 0)
            break;
        if (bytes < 0) {
            if (errno == EINTR)
                continue;
            int saved_errno = errno;
            close(fd);
            errno = saved_errno;
            return -1;
        }
        used += (size_t)bytes;
    }
    close(fd);
    if (used == max_bytes) {
        errno = EOVERFLOW;
        return -1;
    }
    return (int)used;
}

static int save_bytes(int dir, const char *name, const void *data, size_t length)
{
    int fd = openat(dir, name, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
        perror(name);
        return -1;
    }
    size_t written = 0;
    while (written < length) {
        ssize_t bytes = write(fd, (const unsigned char *)data + written, length - written);
        if (bytes < 0 && errno == EINTR)
            continue;
        if (bytes <= 0)
            break;
        written += (size_t)bytes;
    }
    int close_error = close(fd);
    if (written != length || close_error) {
        fprintf(stderr, "%s: incomplete evidence write\n", name);
        return -1;
    }
    return 0;
}

static int check_driver(int dir)
{
    static const char driver[] = "fzspeech-first-embedding";
    static const char compatible[] = "edgespeech,fzspeech-first-embedding-v2";
    char target[256], property[128];
    struct statfs filesystem;
    if (fstatfs(dir, &filesystem) || filesystem.f_type != SYSFS_MAGIC)
        return -1;
    ssize_t link_bytes = readlinkat(dir, "driver", target, sizeof(target) - 1);
    if (link_bytes <= 0 || (size_t)link_bytes == sizeof(target) - 1)
        return -1;
    target[link_bytes] = '\0';
    const char *name = strrchr(target, '/');
    int property_bytes = read_node(dir, "of_node/compatible", property, sizeof(property));
    if (!name || strcmp(name + 1, driver) || property_bytes != sizeof(compatible) ||
        memcmp(property, compatible, sizeof(compatible)))
        return -1;
    return 0;
}

static int terminal(const char *phase)
{
    return !strcmp(phase, "complete\n") || !strcmp(phase, "failed\n") ||
           !strcmp(phase, "failed-to-start\n") || !strcmp(phase, "quarantined\n");
}

static int get_phase(int dir, char phase[64])
{
    int bytes = read_node(dir, "phase", phase, 63);
    if (bytes <= 0 || phase[bytes - 1] != '\n') {
        phase[0] = '\0';
        return -1;
    }
    phase[bytes] = '\0';
    return bytes;
}

static int wait_terminal(int dir, char phase[64])
{
    struct timespec started, now, interval = {.tv_nsec = 200000000};
    if (clock_gettime(CLOCK_MONOTONIC, &started))
        return -1;
    do {
        int length = get_phase(dir, phase);
        if (length < 0 || terminal(phase))
            return length;
        if (strcmp(phase, "idle\n") && strcmp(phase, "starting\n") && strcmp(phase, "ready\n") &&
            strcmp(phase, "waiting-hp0-width\n") &&
            strcmp(phase, "running\n") && strcmp(phase, "draining\n"))
            return -1;
        nanosleep(&interval, NULL);
        if (clock_gettime(CLOCK_MONOTONIC, &now))
            return -1;
    } while (now.tv_sec - started.tv_sec < 120);
    fprintf(stderr, "timed out observing Embedding phase\n");
    return -1;
}

static int submit_once(int dir)
{
    static const char command[] = "embed\n";
    int fd = openat(dir, "embed", O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0)
        return -1;
    /* An interrupted/short write has unknown admission. Never resubmit. */
    ssize_t bytes = write(fd, command, sizeof(command) - 1);
    int saved_errno = errno;
    int close_error = close(fd);
    if (bytes < 0)
        errno = saved_errno;
    else if (bytes != sizeof(command) - 1)
        errno = EIO;
    return bytes == sizeof(command) - 1 && !close_error ? 0 : -1;
}

static int export_evidence(int node, int out, int complete)
{
    static const struct {
        const char *attr, *filename;
        size_t min_bytes, max_bytes;
    } evidence[] = {
        {"result", "result.txt", 1, 4096},
        {"embedding_packet", "packet.bin", FZ_JOB_HEADER_BYTES, FZ_JOB_MAX_PACKET_BYTES},
        {"embedding_output", "output.f32le", 16384, 16384},
        {"embedding_guards_before", "guards.before.bin", 16384, 16384},
        {"embedding_guards_after", "guards.after.bin", 16384, 16384},
        {"embedding_registers", "registers.bin", 3 * 0x570, 3 * 0x570},
    };
    unsigned char data[16385];
    int status = 0;
    for (size_t i = 0; i < sizeof(evidence) / sizeof(evidence[0]); ++i) {
        int bytes;
        /* The driver publishes terminal phase immediately before unlocking
         * the receipt. Restart only read-side operations if its lock is busy. */
        for (unsigned attempt = 0; ; ++attempt) {
            bytes = read_node(node, evidence[i].attr, data, evidence[i].max_bytes + 1);
            if (bytes >= 0 || errno != EAGAIN || attempt == 10)
                break;
            const struct timespec delay = {.tv_nsec = 20000000};
            nanosleep(&delay, NULL);
        }
        if (bytes < 0 || (size_t)bytes < evidence[i].min_bytes) {
            if (complete || i == 0) {
                fprintf(stderr, "%s: missing or incomplete evidence\n", evidence[i].attr);
                status = -1;
            }
            continue;
        }
        if (save_bytes(out, evidence[i].filename, data, (size_t)bytes))
            status = -1;
    }
    return status;
}

static int board_diagnostic(const char *path, const char *destination, int execute)
{
    static const char boot_id[] = "/proc/sys/kernel/random/boot_id";
    char boot_before[64], boot_after[64], phase[64];
    int result = EXIT_FAILURE;
    int node = open(path, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (node < 0 || check_driver(node)) {
        fprintf(stderr, "%s: expected a bound Embedding sysfs node\n", path);
        if (node >= 0)
            close(node);
        return result;
    }
    if (mkdir(destination, 0700)) {
        perror(destination);
        close(node);
        return result;
    }
    int out = open(destination, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (out < 0) {
        perror(destination);
        close(node);
        return result;
    }
    int before_bytes = read_node(AT_FDCWD, boot_id, boot_before, sizeof(boot_before));
    int phase_bytes = get_phase(node, phase);
    if (before_bytes != 37 || boot_before[36] != '\n' || phase_bytes < 0 ||
        save_bytes(out, "boot_id.before", boot_before, (size_t)before_bytes) ||
        save_bytes(out, "phase.before", phase, (size_t)phase_bytes))
        goto done;

    int admitted = 0;
    if (execute && strcmp(phase, "idle\n")) {
        fprintf(stderr, "requires idle phase for the single submission\n");
    } else if (execute) {
        admitted = submit_once(node) == 0;
        if (!admitted)
            perror("embed");
    }
    if (admitted)
        phase_bytes = wait_terminal(node, phase);
    if (phase_bytes < 0)
        phase_bytes = get_phase(node, phase);
    int complete = phase_bytes > 0 && !strcmp(phase, "complete\n");
    int phase_saved = phase_bytes > 0 && !save_bytes(out, "phase.after", phase, (size_t)phase_bytes);
    int exported = export_evidence(node, out, complete) == 0;
    int after_bytes = read_node(AT_FDCWD, boot_id, boot_after, sizeof(boot_after));
    int same_boot = after_bytes == before_bytes && !memcmp(boot_before, boot_after, (size_t)before_bytes);
    int boot_saved = after_bytes > 0 && !save_bytes(out, "boot_id.after", boot_after, (size_t)after_bytes);
    if ((!execute || admitted) && complete && phase_saved && exported && same_boot && boot_saved)
        result = EXIT_SUCCESS;
    printf("{\"mode\":\"%s\",\"complete_evidence_exported\":%s}\n",
           execute ? "single_submission" : "collect_only", result == EXIT_SUCCESS ? "true" : "false");

done:
    close(out);
    close(node);
    return result;
}
#endif

int main(int argc, char **argv)
{
    int board = 0;
#ifdef __linux__
    board = argc == 5 && (!strcmp(argv[2], "--execute") || !strcmp(argv[2], "--collect"));
#endif
    if (argc != 2 && !board) {
        fprintf(stderr, "usage: %s FIXTURE [--execute|--collect SYSFS_NODE NEW_EXPORT_DIR]\n", argv[0]);
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < sizeof(assets) / sizeof(assets[0]); ++i) {
        if (verify_asset(argv[1], &assets[i]))
            return EXIT_FAILURE;
    }
#ifdef __linux__
    if (board)
        return board_diagnostic(argv[3], argv[4], !strcmp(argv[2], "--execute"));
#endif
    puts("{\"mode\":\"offline_plan_only\",\"inputs\":4,\"reference_elements\":4096}");
    return EXIT_SUCCESS;
}
