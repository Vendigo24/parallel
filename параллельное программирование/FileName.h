#pragma once
#include <concepts>
#include <stdint.h>
#include <thread>
#include "my_thread.h"



template <class T, std::unsigned_integral U>
auto my_pow(T x, U n) {
	T r = T(1);
	while (n > 0) {
		if (n & 1)
			r *= x;
		x *= x;
		n >>= 1;
	}
	return r;
};

template <class T, std::unsigned_integral U> requires requires (T a) { a *= a; }
auto my_pow(T x, U n) {
	T r;
	while (n > 0) {
		if (n & 1)
			r *= x;
		x *= x;
		n >>= 1;
	}
	return r;
};

// Рандомизация данных при помощи линейного конгруетного генератора



class lc_t {
	uint32_t A, B;
public:
	lc_t(uint32_t a = 1, uint32_t b = 0) : A(a), B(b) {}

	lc_t operator *= (const lc_t& x){
		B += A * x.B;
		A *= x.A;
		return *this;
	}

	auto operator()(uint32_t seed) const {
		return A * seed + B;
	}

	auto operator()(uint32_t seed, uint32_t min_val, uint32_t max_val) const {
		return !(max_val - min_val + 1) ? (*this)(seed) : (*this)(seed) % (max_val - min_val + 1) + min_val;
	}
};

double randomize_vector(uint32_t* V, size_t n, uint32_t seed, uint32_t min_val = 0, uint32_t max_val = UINT32_MAX) {
	double res = 0;
	uint32_t A = 22695477;
	uint32_t B = 1;
	lc_t g = lc_t(A, B);
	lc_t curr = lc_t(A, B);
	for (int i = 0; i < n; ++i) {
		curr *= g;
		V[i] = curr(seed, min_val, max_val);
		res += V[i];
	}
	return res / n;
}

double randomize_vector_parallel(uint32_t* V, size_t n, uint32_t seed, uint32_t min_val = 0, uint32_t max_val = UINT32_MAX) {
	if (min_val > max_val) { exit(__LINE__); }
	double res = 0;
	uint32_t A = 22695477;
	uint32_t B = 1;
	std::vector<std::thread> workers;
	std::mutex mtx;
	unsigned T = get_num_threads();
	auto worker_proc = [V, n, min_val, max_val, seed, A, B, T, &res, &mtx](unsigned t) {
		uint32_t part_res = 0;
		lc_t g = lc_t(A, B);
		size_t b = n % T, e = n / T;
		if (t < b) b = t * ++e;
		else b += t * e;
		e += b;
		lc_t curr_g = lc_t();
		curr_g = my_pow(g, b + 1);
		for (int i = b; i < e; ++i) {
			curr_g *= g;
			V[i] = curr_g(seed, min_val, max_val);
			part_res += V[i];
		}
		std::scoped_lock{ mtx };
		res += part_res;
	};

	for (unsigned t = 1; t < T; ++t)
		workers.emplace_back(worker_proc, t);
	worker_proc(0);
	for (auto& w : workers)
		w.join();
	return res / n;
}

double randomize_vector_par(std::vector<uint32_t>& v, uint32_t seed, uint32_t min_val = 0, uint32_t max_val = UINT32_MAX) {
	return randomize_vector_parallel(v.data(), v.size(), seed, min_val, max_val);
}

double randomize_vector(std::vector<uint32_t>& v, uint32_t seed, uint32_t min_val = 0, uint32_t max_val = UINT32_MAX) {
	return randomize_vector(v.data(), v.size(), seed, min_val, max_val);
}

bool randomize_test() {
	std::vector<uint32_t>v1(10), v2(10);
	randomize_vector(v1, 0, 0, 5);
	randomize_vector_par(v2, 0, 0, 5);
	for (auto& i : v1) {
		std::cout << i<<" ";
	}
	std::cout << "\n";
	for (auto& i : v2) {
		std::cout << i << " ";
	}
	std::cout << "\n";
	if (randomize_vector(v1, 0, 0, 5) != randomize_vector_par(v2, 0,0, 5))
		return false;
	auto pr = std::ranges::mismatch(v1, v2);
	return pr.in1 == v1.end() && pr.in2 == v2.end();
}
