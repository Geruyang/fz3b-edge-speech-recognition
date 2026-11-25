/* Fixed live ABI for the single 200 MHz layer on the 30-bit HP0 port. */
#ifndef FZ_LAYER_JOB_H
#define FZ_LAYER_JOB_H
#include <linux/types.h>
#include <linux/ioctl.h>
#define LAYER_BOUNDARY 8200
#define LAYER_HISTORY 43272
struct layer_job {
    __u32 layer, batch;
    __u64 input, history, output, next_history;
    __u64 sequence, copy_in_ns, setup_ns, start_irq_ns, irq_wake_ns, copy_out_ns;
    __u32 control, result;
};
#define LAYER_RUN _IOWR('L', 1, struct layer_job)
#endif
