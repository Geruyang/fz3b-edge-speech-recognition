/* First Embedding diagnostic only. The maintenance owner holds the existing
 * service/loader locks. This companion handles the driver's width window. */
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
    if (words[offsets[0]] != 0x3b1 || words[offsets[1]] != 0x3b1) return 2;
    for (unsigned n = 0; n < 2; ++n) {
        unsigned offset = offsets[n];
        if (!in_window(boot) || words[offset] != 0x3b1) return 3;
        words[offset] = 0x3b0;
#ifdef __aarch64__
        __asm__ volatile ("dsb sy" ::: "memory");
#endif
        uint32_t value = words[offset];
        printf("HP0 offset=%#x before=0x3b1 expected=0x3b0 readback=%#x\n",
               (unsigned)(offset * sizeof(uint32_t)), value);
        if (value != 0x3b0) return 4;
    }
    return 0;
}

#ifdef FZ_WIDTH_TEST
/* Host test for the two writes, including loss of the window between them. */
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
            words[5] != (limit > 1 ? 0x3b0U : 0x3b1U)) return 1;
        for (unsigned i = 1; i < 5; ++i) if (words[i] != i) return 1;
    }
    uint32_t mixed[] = {0x3b0, 0, 0, 0, 0, 0x3b1};
    remaining = 2;
    if (apply_width(mixed, "test", simulate_window) != 2 || mixed[5] != 0x3b1 || remaining != 2)
        return 1;
    puts("PASS: only two intended words, lost window, mixed initial widths");
    return 0;
}
#else
static int line_matches(const char *path, const char *expected)
{
    char line[100];
    FILE *file = fopen(path, "r");
    if (!file) return 0;
    int ok = fgets(line, sizeof(line), file) != NULL;
    if (ok) {
        size_t n = strlen(line);
        ok = n && line[n-1] == '\n';
        if (ok) { line[n-1] = '\0'; ok = strcmp(line, expected) == 0; }
    }
    fclose(file);
    return ok;
}

static int in_window(const char *boot)
{
    return line_matches("/proc/sys/kernel/random/boot_id", boot) &&
           line_matches("/sys/bus/platform/drivers/fzspeech-first-embedding/a0000000.fzspeech/phase",
                        "waiting-hp0-width");
}

int main(int argc, char **argv)
{
    setbuf(stdout, NULL);
    if (argc != 3 || strcmp(argv[1], "--boot-id") || strlen(argv[2]) != 36 || !in_window(argv[2]))
        return 1;
    int fd = open("/dev/mem", O_RDWR | O_SYNC | O_CLOEXEC);
    if (fd < 0) return 5;
    void *page = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0xfd380000LL);
    if (page == MAP_FAILED) { close(fd); return 6; }
    int rc = apply_width(page, argv[2], in_window);
    munmap(page, 4096);
    close(fd);
    return rc;
}
#endif
