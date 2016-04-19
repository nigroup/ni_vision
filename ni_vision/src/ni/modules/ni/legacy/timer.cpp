#include "ni/legacy/timer.h"

#include <time.h>

int64_t timespecDiff (struct timespec *timeA_p, struct timespec *timeB_p) {
  return ((timeA_p->tv_sec * 1000000000) + timeA_p->tv_nsec) - ((timeB_p->tv_sec * 1000000000) + timeB_p->tv_nsec);
}
