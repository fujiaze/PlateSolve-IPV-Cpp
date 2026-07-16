#ifndef PSOLVE_LOG_H
#define PSOLVE_LOG_H

typedef enum {
    PSOLVE_LOG_DEBUG = 0,
    PSOLVE_LOG_INFO  = 1,
    PSOLVE_LOG_WARN  = 2,
    PSOLVE_LOG_ERROR = 3
} PSolveLogLevel;

void psolve_log_init(const char *log_dir);
void psolve_log_close(void);
void psolve_log_write(PSolveLogLevel level, const char *fmt, ...);

#define PSLOG_D(fmt, ...) psolve_log_write(PSOLVE_LOG_DEBUG, fmt, ##__VA_ARGS__)
#define PSLOG_I(fmt, ...) psolve_log_write(PSOLVE_LOG_INFO,  fmt, ##__VA_ARGS__)
#define PSLOG_W(fmt, ...) psolve_log_write(PSOLVE_LOG_WARN,  fmt, ##__VA_ARGS__)
#define PSLOG_E(fmt, ...) psolve_log_write(PSOLVE_LOG_ERROR, fmt, ##__VA_ARGS__)

#endif
