#include "psolve_log.h"
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/time.h>
#endif

static FILE *g_log_fp = NULL;
static PSolveLogLevel g_log_level = PSOLVE_LOG_DEBUG;

static const char *level_names[] = {"DEBUG", "INFO", "WARN", "ERROR"};

void psolve_log_init(const char *log_dir) {
    if (g_log_fp) return;
    char path[1024];
#ifdef _WIN32
    snprintf(path, sizeof(path), "%s\\plate_solve.log", log_dir);
#else
    snprintf(path, sizeof(path), "%s/plate_solve.log", log_dir);
#endif
    g_log_fp = fopen(path, "a");
    if (!g_log_fp) {
        fprintf(stderr, "[plate_solve] cannot open log: %s\n", path);
    }
}

void psolve_log_close(void) {
    if (g_log_fp) {
        fclose(g_log_fp);
        g_log_fp = NULL;
    }
}

void psolve_log_write(PSolveLogLevel level, const char *fmt, ...) {
    if (level < g_log_level) return;
    if (!g_log_fp) return;

    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", t);

    va_list ap;
    va_start(ap, fmt);

    fprintf(g_log_fp, "[%s] [%s] ", ts, level_names[level]);
    vfprintf(g_log_fp, fmt, ap);
    fprintf(g_log_fp, "\n");
    fflush(g_log_fp);

    if (level >= PSOLVE_LOG_WARN) {
        fprintf(stderr, "[plate_solve] [%s] ", level_names[level]);
        va_end(ap);
        va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        fprintf(stderr, "\n");
    }

    va_end(ap);
}
