#pragma once
#include <iostream>
#include <functional>
#include <memory>
#include <omp.h>
#include <thread>
#include <mutex>
#include <vector>
#include <queue>
#include <chrono>
#include "my_thread.h"

double get_omp_time(double (*f)(const double*, size_t), const double* v, size_t n) {
	auto t1 = omp_get_wtime();
	f(v, n);
	return omp_get_wtime(); -t1;
}

template <std::invocable<const double*, size_t>F>
auto get_cpp_time(F f, const double* v, size_t n) {

	using namespace std::chrono;
	auto t1 = steady_clock.now();
	f(v, n);
	auto t2 = steady_clock::now();
	return duration_cast<milliseconds>(t2 - t1).count();
}
struct prof_res_t {
	double res;
	double time, speedup, efficiency;
	unsigned T;
};

prof_res_t* run_omp_experiment(double (*f)(const double* V, size_t n), const double* V, size_t n) {
	prof_res_t* res_table = (prof_res_t*)malloc(omp_get_num_procs() * sizeof(struct prof_res_t));

	for (unsigned T = 1; T <= omp_get_num_procs(); ++T) {
		omp_set_num_threads(T);
		auto t1 = omp_get_wtime();
		res_table[T - 1].res = f(V, n);
		auto t2 = omp_get_wtime();
		res_table[T - 1].time = t2 - t1;
		res_table[T - 1].speedup = res_table[0].time / res_table[T - 1].time;
		res_table[T - 1].efficiency = res_table[T - 1].speedup / T;
		res_table[T - 1].T = T;
	};
	return res_table;
}

std::vector<prof_res_t> run_cpp_experiment(std::function<double(const double* V, size_t n)> f, const double* V, size_t n) {
	std::vector<prof_res_t> res_table;
	std::size_t T_max = get_num_threads();
	for (std::size_t T = 1; T <= T_max; ++T) {
		using namespace std::chrono;
		set_num_threads(T);
		prof_res_t rr;
		rr.T = T;
		auto t0 = steady_clock::now();
		rr.res = f(V, n);
		auto t1 = steady_clock::now();
		rr.time = duration_cast<milliseconds>(t1 - t0).count();
		res_table.emplace_back(rr);
		res_table[T - 1].speedup = res_table[0].time / res_table[T - 1].time;
		res_table[T - 1].efficiency = res_table[T - 1].speedup / T;
	}
	return res_table;
}

std::vector<prof_res_t> run_cpp_experiment_generate(double (*f)(std::vector<uint32_t>& v, uint32_t seed, uint32_t min_val, uint32_t max_val), std::vector<uint32_t> V, uint32_t seed, uint32_t min_val=0, uint32_t max_val= UINT32_MAX) {
	std::vector<prof_res_t> res_table;
	std::size_t T_max = get_num_threads();
	for (std::size_t T = 1; T <= T_max; ++T) {
		set_num_threads(T);
		prof_res_t rr;
		rr.T = T;
		auto t0 = omp_get_wtime();
		rr.res = f(V, seed, min_val, max_val);
		auto t1 = omp_get_wtime();
		rr.time = (t1 - t0);
		res_table.emplace_back(rr);
		res_table[T - 1].speedup = res_table[0].time / res_table[T - 1].time;
		res_table[T - 1].efficiency = res_table[T - 1].speedup / T;
	}
	return res_table;
}