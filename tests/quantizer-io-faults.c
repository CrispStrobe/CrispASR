// Linux-only stdio fault injection for actual quantizer execution tests.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE* output;
static FILE* input;
static int writes;
static int flushes;

static int fail(const char* stage) {
    const char* requested = getenv("QUANTIZER_IO_FAULT");
    if (!requested || strcmp(requested, stage) != 0)
        return 0;
    fprintf(stderr, "injected %s failure\n", stage);
    errno = ENOSPC;
    return 1;
}

FILE* fopen(const char* path, const char* mode) {
    FILE* (*real_open)(const char*, const char*) = dlsym(RTLD_NEXT, "fopen");
    FILE* result = real_open(path, mode);
    const char* target = getenv("QUANTIZER_IO_OUTPUT");
    if (target && strcmp(target, path) == 0 && strchr(mode, 'w'))
        output = result;
    const char* source = getenv("QUANTIZER_IO_INPUT");
    if (source && strcmp(source, path) == 0 && strchr(mode, 'r'))
        input = result;
    return result;
}

int fseeko(FILE* stream, off_t offset, int whence) {
    int (*real_seek)(FILE*, off_t, int) = dlsym(RTLD_NEXT, "fseeko");
    if (output && stream == input && fail("input-seek"))
        return -1;
    return real_seek(stream, offset, whence);
}

size_t fwrite(const void* data, size_t size, size_t count, FILE* stream) {
    size_t (*real_write)(const void*, size_t, size_t, FILE*) = dlsym(RTLD_NEXT, "fwrite");
    if (stream == output) {
        ++writes;
        const char* stage = writes == 1 ? "placeholder"
                            : (size * count >= 4 && memcmp(data, "GGUF", 4) == 0)
                                ? "metadata"
                                : (size * count < 32 ? "padding" : "tensor");
        if (fail(stage)) {
            // A partial write matters too: the caller must reject a short count.
            return real_write(data, size, count / 2, stream);
        }
    }
    return real_write(data, size, count, stream);
}

int fputc(int ch, FILE* stream) {
    int (*real_putc)(int, FILE*) = dlsym(RTLD_NEXT, "fputc");
    if (stream == output && fail("padding"))
        return EOF;
    return real_putc(ch, stream);
}

int fflush(FILE* stream) {
    int (*real_flush)(FILE*) = dlsym(RTLD_NEXT, "fflush");
    if (stream == output && fail(++flushes == 1 ? "data-flush" : "metadata-flush"))
        return EOF;
    return real_flush(stream);
}

int fseek(FILE* stream, long offset, int whence) {
    int (*real_seek)(FILE*, long, int) = dlsym(RTLD_NEXT, "fseek");
    if (stream == output && fail("metadata-seek"))
        return -1;
    return real_seek(stream, offset, whence);
}

int fclose(FILE* stream) {
    int (*real_close)(FILE*) = dlsym(RTLD_NEXT, "fclose");
    int is_output = stream == output;
    int result = real_close(stream);
    if (is_output) {
        output = NULL;
        if (fail("close"))
            return EOF;
    }
    return result;
}
