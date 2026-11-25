/* SPDX-License-Identifier: GPL-2.0 */
#ifndef FZSPEECH_PUBLICATION_H
#define FZSPEECH_PUBLICATION_H

#include "fzspeech_transport_regs.h"

/* Process-context, serialized provider control. The sequencer admission gate
 * rejects new doorbells but never cancels an accepted invocation. Physical RUN
 * remains enabled until the invocation and every physical port have retired.
 */
struct fz_publication_control {
    void *context;
    int (*set_accept)(void *context, fz_u32 enabled);
    int (*set_run)(void *context, fz_u32 enabled);
    int (*observe)(void *context, struct fz_transport_stats *stats);
    fz_u32 (*get_run)(void *context);
    fz_u32 desired_run;
};

static inline int fz_pub_control_set_run(struct fz_publication_control *control, fz_u32 enabled)
{
    int result;
    control->desired_run = 0;
    if (!enabled)
        return control->set_accept(control->context, 0) ? FZ_TR_IO_ERROR : FZ_TR_OK;
    result = control->set_run(control->context, 1);
    if (!result)
        result = control->set_accept(control->context, 1);
    if (result) {
        /* A failed enable may have taken effect. Stop admission, retain RUN
         * and all DMA storage; the caller treats this as ambiguous. */
        control->set_accept(control->context, 0);
        return FZ_TR_IO_ERROR;
    }
    control->desired_run = 1;
    return FZ_TR_OK;
}

static inline int fz_pub_control_snapshot(struct fz_publication_control *control,
                                         struct fz_transport_stats *stats)
{
	fz_u32 physical_run;
    if (control->observe(control->context, stats))
        return FZ_TR_IO_ERROR;
    if (!control->desired_run && fz_tr_retired(stats)) {
        /* issuance_stopped may mean admission=0 while physical RUN is still
         * one. Only the actual RUN register can authorize SETGEN/profile. */
        physical_run = control->get_run(control->context);
        if (physical_run > 1)
            return FZ_TR_IO_ERROR;
        if (physical_run) {
            if (control->set_run(control->context, 0) || control->get_run(control->context) != 0 ||
                control->observe(control->context, stats))
                return FZ_TR_IO_ERROR;
        }
        if (!fz_tr_drained(stats)) return FZ_TR_IO_ERROR;
    }
    return FZ_TR_OK;
}

#endif
