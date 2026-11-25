#include <cstdint>
#include <iostream>

#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <dlfcn.h>
#include <gnu/libc-version.h>
#include <sys/auxv.h>
#endif

int main(int argc, char** argv) {
    if (sizeof(void*) != 8 || sizeof(std::int64_t) != 8) {
        std::cerr << "A 64-bit ABI is required\n";
        return 1;
    }
#if defined(__linux__) && defined(__aarch64__)
    const unsigned long hwcap = getauxval(AT_HWCAP);
    if ((hwcap & (HWCAP_FP | HWCAP_ASIMD)) != (HWCAP_FP | HWCAP_ASIMD)) {
        std::cerr << "Missing AArch64 FP or ASIMD\n";
        return 2;
    }
    std::cout << "aarch64 fp asimd glibc=" << gnu_get_libc_version() << '\n';
    // Callers pass absolute paths to their explicitly selected private libraries.
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] != '/') {
            std::cerr << "Library path must be absolute\n";
            return 3;
        }
        void* library = dlopen(argv[i], RTLD_NOW | RTLD_LOCAL);
        if (library == nullptr) {
            std::cerr << argv[i] << ": " << dlerror() << '\n';
            return 4;
        }
        std::cout << "loaded " << argv[i] << '\n';
        dlclose(library);
    }
#else
    (void)argv;
    if (argc != 1) {
        std::cerr << "Private library probe requires Linux AArch64\n";
        return 3;
    }
    std::cout << "host 64-bit C++17 executable started\n";
#endif
    return 0;
}
