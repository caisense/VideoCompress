/*
 * ABI bridge for the RK3588 Buildroot image (glibc 2.37).
 *
 * The installed cross SDK uses glibc 2.38 headers, which redirect strtol to
 * the new C23 entry point __isoc23_strtol.  The board deliberately remains on
 * glibc 2.37, whose exported legacy strtol ABI has the semantics we need for
 * command-line integer parsing.  Define the new symbol inside the executable
 * and bind its implementation explicitly to that legacy exported ABI.
 *
 * Do not include <stdlib.h>: doing so would apply the same redirect to this
 * bridge and reintroduce the unresolved GLIBC_2.38 dependency.
 */

#if defined(__GNUC__)
#define ROI_HIDDEN __attribute__((visibility("hidden")))
#else
#define ROI_HIDDEN
#endif

extern long roi_legacy_strtol(const char *nptr, char **endptr,
                              int base) __asm__("strtol");

ROI_HIDDEN long __isoc23_strtol(const char *nptr, char **endptr, int base)
{
    return roi_legacy_strtol(nptr, endptr, base);
}
