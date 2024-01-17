#pragma once
#include <thread>
#include <omp.h>

static unsigned g_thread_num = 12;
unsigned get_num_threads() {
	return g_thread_num;
}

void set_num_threads(unsigned T) {
	g_thread_num = T;
	omp_set_num_threads(T);
}