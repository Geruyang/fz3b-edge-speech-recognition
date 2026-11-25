/* Switch only the HP0 width fields in the active stream window. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifndef FZ_WIDTH_TEST
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

typedef int (*window_check)(const char *boot);

static int apply_width(volatile uint32_t *words, const char *boot, window_check in_window)
{
    const unsigned offsets[] = {0, 0x14 / sizeof(uint32_t)};
    if (words[offsets[0]] != 0x3b1 || words[offsets[1]] != 0x3b1)
        return 2;
    for (unsigned n = 0; n < 2; ++n) {
        unsigned offset = offsets[n];
        if (!in_window(boot) || words[offset] != 0x3b1)
            return 3;
        words[offset] = 0x3b0;
#ifdef __aarch64__
        __asm__ volatile ("dsb sy" ::: "memory");
#endif
        uint32_t value = words[offset];
        printf("HP0 offset=%#x before=0x3b1 expected=0x3b0 readback=%#x\n",
               (unsigned)(offset * sizeof(uint32_t)), value);
        if (value != 0x3b0)
            return 4;
    }
    return words[offsets[0]] == 0x3b0 && words[offsets[1]] == 0x3b0 ? 0 : 4;
}

#ifdef FZ_WIDTH_TEST
static int remaining;
static int simulate_window(const char *boot)
{
    return !strcmp(boot, "test") && remaining-- > 0;
}

int main(void)
{
    for (int limit = 0; limit <= 2; ++limit) {
        uint32_t words[] = {0x3b1, 1, 2, 3, 4, 0x3b1};
        remaining = limit;
        int rc = apply_width(words, "test", simulate_window);
        if (rc != (limit == 2 ? 0 : 3) || words[0] != (limit > 0 ? 0x3b0U : 0x3b1U) ||
            words[5] != (limit > 1 ? 0x3b0U : 0x3b1U))
            return 1;
        for (unsigned i = 1; i < 5; ++i)
            if (words[i] != i)
                return 1;
    }
    uint32_t mixed[] = {0x3b0, 0, 0, 0, 0, 0x3b1};
    remaining = 2;
    if (apply_width(mixed, "test", simulate_window) != 2 || mixed[5] != 0x3b1 || remaining != 2)
        return 1;
    puts("PASS: two intended words, no write outside the stream phase, no write on a mixed initial width");
    return 0;
}
#else
#define STREAM_DRIVER "/sys/bus/platform/drivers/fzspeech-embedding-stream"
#define STREAM_NODE STREAM_DRIVER "/a0000000.fzspeech"

static int file_matches(const char *path, const void *expected, size_t n)
{
    unsigned char data[100];
    FILE *file = fopen(path, "rb");
    if (!file)
        return 0;
    size_t bytes = fread(data, 1, sizeof(data), file);
    fclose(file);
    return bytes == n && !memcmp(data, expected, n);
}

static int in_window(const char *boot)
{
    static const char phase[] = "waiting-hp0-width\n";
    static const char compatible[] = "edgespeech,fzspeech-embedding-stream-v2";
    static const char module[] = "/fzspeech_of_embedding_stream";
    char actual[128], boot_line[38];
    int n = snprintf(boot_line, sizeof(boot_line), "%s\n", boot);
    if (n != 37 || !file_matches("/proc/sys/kernel/random/boot_id", boot_line, (size_t)n))
        return 0;
    ssize_t got = readlink(STREAM_DRIVER "/module", actual, sizeof(actual));
    size_t m = sizeof(module) - 1;
    if (got < (ssize_t)m || got == (ssize_t)sizeof(actual) || memcmp(actual + got - m, module, m))
        return 0;
    return file_matches(STREAM_NODE "/of_node/compatible", compatible, sizeof(compatible)) &&
        file_matches(STREAM_NODE "/phase", phase, sizeof(phase) - 1);
}

int main(int argc, char **argv)
{
    setbuf(stdout, NULL);
    if (argc != 3 || strcmp(argv[1], "--boot-id") || strlen(argv[2]) != 36 || !in_window(argv[2]))
        return 1;
    int fd = open("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC);
    if (fd < 0)
        return 5;
    void *page = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0xfd380000LL);
    if (page == MAP_FAILED) {
        close(fd);
        return 6;
    }
    int rc = apply_width(page, argv[2], in_window);
    munmap(page, 4096);
    close(fd);
    return rc;
}
#endif
