#include <iostream>
#include <functional>
#include <memory>
#include <omp.h>
#include <io.h>
#include <thread>
#include <mutex>
#include <vector>
#include <queue>
#include <chrono>
#include "experiment.h"
#include "my_thread.h"
#include "my_barrier.h"
#include "FileName.h"

/*Round Robin start*/
struct partial_sum_t1 {
	alignas(64) double value;
};

struct partial_sum_t2 {
	union {
		double value;
		char padd[64];
	};
};

double average(const double* v, size_t n) {
	double res = 0.0;
	for (size_t i = 0; i < n; i++) res += v[i];
	return res / n;
}

double average_reduce(const double* v, size_t n) {
	double res = 0.0;
	#pragma omp parallel for reduction(+: res)
	for (int i = 0; i < n; i++) res += v[i];
	return res / n;
}

double average_rr(const double* v, size_t n) {
	double res = 0.0;
#pragma omp parallel
	{
	unsigned t = omp_get_thread_num();
	unsigned T = omp_get_num_threads();
	for (int i = t; i < n; i += T) res += v[i];
	}
	return res / n;
}

double average_omp(const double* v, size_t n) {
	double res = 0.0, *partial_sums = (double*)calloc(omp_get_num_procs(), sizeof(double));
#pragma omp parallel
	{
		unsigned t = omp_get_thread_num();
		unsigned T = omp_get_max_threads();
		for (int i = t; i < n; i += T) partial_sums[t] += v[i];
	}
	for (size_t i = 1; i < omp_get_num_procs(); ++i)
		partial_sums[0] += partial_sums[i];
	res = partial_sums[0] / n;
	free(partial_sums);
	return res;
}

double average_omp_modified(const double* v, size_t n) {
	unsigned T;
	double res = 0.0;
	partial_sum_t2*partial_sums;
#pragma omp parallel
	{
		unsigned t = omp_get_thread_num();
		
#pragma omp single
		{
			T = omp_get_max_threads();
		partial_sums = (partial_sum_t2*) malloc(T * sizeof partial_sum_t2);
		}
		partial_sums[t].value = 0.0;
		for (int i = t; i < n; i += T) partial_sums[t].value += v[i];
	}
	for (size_t i = 1; i < omp_get_num_procs(); ++i)
		partial_sums[0].value += partial_sums[i].value;
	res = partial_sums[0].value / n;
	free(partial_sums);
	return res;
}

double average_omp_mtx(const double* v, size_t n) {
	double res = 0.0;
#pragma omp parallel
	{
		unsigned T = omp_get_num_threads();
		unsigned t = omp_get_thread_num();
		for (int i = t; i < n; i += T) {
#pragma omp critical
			{
				res += v[i];
			}
		}
	}
	return res / n;
}

double average_omp_mtx_modified(const double* v, size_t n) {
	double res = 0.0;
	size_t T, t;
#pragma omp parallel
	{
		double partial_sum = 0.0;
		T = omp_get_max_threads();
		t = omp_get_thread_num();
		for (size_t i = t; i < n; i += T) 
			partial_sum += v[i];
			#pragma omp critical
			{
				res += partial_sum;
			}
	}
	return res / n;
}

double average_cpp_mtx(const double* v, size_t n) {
	unsigned T = get_num_threads();
	double res = 0.0;
	std::vector<std::thread> workers;
	std::mutex mtx;
	auto worker_proc = [&mtx, T, v, n, &res](unsigned t) {
		double partial_result = 0.0;
		for (std::size_t i = t; i < n; i += T)
			partial_result += v[i];
		mtx.lock();
		res += partial_result;
		mtx.unlock();
	};
	for (unsigned t = 0; t < T; ++t)
		workers.emplace_back(worker_proc, t);
	for (auto& w : workers)
		w.join();
	return res / n;
	}

double average_cpp_mtx_modified0(const double* v, size_t n) {
	unsigned T = std::thread::hardware_concurrency();
	double res = 0.0;
	std::vector<std::thread> workers;
	std::mutex mtx;
	auto worker_proc = [&mtx,&res, T, v, n ](unsigned t) {
		double partial_result = 0.0;
		for (std::size_t i = t; i < n; i += T)
			partial_result += v[i];
		mtx.lock();
		res += partial_result;
		mtx.unlock();
	};
	for (unsigned int t = 1; t < T; ++t)
		workers.emplace_back(worker_proc, t);
	worker_proc(0);
	for (auto& w : workers)
		w.join();
	return res / n;
}

double average_cpp_mtx_modified1(const double* v, size_t n) {
	size_t T = std::thread::hardware_concurrency();
	double res = 0.0;
	std::vector<std::thread> workers;
	std::mutex mtx;
	auto worker_proc = [&mtx, T, v, n, &res](size_t t) {
		double partial_result = 0.0;
		for (std::size_t i = t; i < n; i += T)
			partial_result += v[i];
		std::scoped_lock l{ mtx };
		res += partial_result;
	};
	for (unsigned int t = 1; t < T; ++t){
		workers.emplace_back(worker_proc, t);
	}
	worker_proc(0);
	for (auto& w : workers)
		w.join();
	return res / n;
}

/*Round Robin end*/

/*Локализация данных start*/

double average_cpp_mtx_local(const double* v, size_t n) {
	double res = 0.0;
	size_t T = get_num_threads();
	std::mutex mtx;
	std::vector<std::thread> workers;
	auto worker_proc = [&mtx, &res, n, T, v](unsigned t) {
		double part_res = 0.0;
		size_t b = n % T, e = n / T;
		if (t < b) b = t * ++e;
		else b += t * e;
		e += b;
		for (auto i = b; i < e; ++i) {
			part_res += v[i];
		}
		mtx.lock();
		res += part_res;
		mtx.unlock();
	};
	for (unsigned int t = 1; t < T; ++t) {
		workers.emplace_back(worker_proc, t);
	}
	worker_proc(0);
	for (auto& w : workers)
		w.join();
	return res / n;
}
/*Локализация данных end*/




/*Синхронизация данных start*/


double average_reduction_local(const double* v, size_t n) {
	//double res = 0.0;
	//size_t T = std::thread::hardware_concurrency();
	//std::mutex mtx;
	//std::vector<std::thread> workers;
	//auto worker_proc = [&mtx, &res, n, T, v](unsigned t) {
	//	double part_res = 0.0;
	//	size_t b = n % T, e = n / T;
	//	if (t < b) b = t * ++e;
	//	else b += t * e;
	//	e += b;
	//	for (auto i = b; i < e; ++i) {
	//		part_res += v[i];
	//	}
	//	mtx.lock();
	//	res += part_res;
	//	mtx.unlock();
	//	};
	//for (unsigned int t = 1; t < T; ++t) {
	//	workers.emplace_back(worker_proc, t);
	//}
	//worker_proc(0);
	//for (auto& w : workers)
	//	w.join();
	//return res / n;
	size_t T = get_num_threads();
	barrier bar(T);
	double* partial_sum = new double[T];
	for (int i = 0; i < T; i++) {
		partial_sum[i] = 0;
	}
	std::vector<std::thread> workers;

	auto worker_proc = [&bar, &partial_sum, v, n, T](unsigned t) {
		double res = 0;
		size_t b = n % T, e = n / T;
		if (t < b) b = t * ++e;
		else b += t * e;
		e += b;
		for (auto i = b; i < e; ++i) {
			res += v[i];
		};
		partial_sum[t] = res;
		for (size_t step = 1, next = 2; step < T; step = next, next += next) {
			bar.arrive_and_wait();
			if ((t & (next - 1)) == 0 && t + step < T) {
				partial_sum[t] += partial_sum[t + step];
			}
		}
	};
	for (unsigned int t = 1; t < T; ++t) {
		workers.emplace_back(worker_proc, t);
	}
	worker_proc(0);
	for (auto& w : workers)
		w.join();
	return partial_sum[0]/n;
}



/*Синхронизация данных end*/

int main() {
	std::size_t N = 1u << 24;
	auto buf = std::make_unique<double[]>(N);
	for (std::size_t i = 0; i < N; ++i) buf[i] = i;

	//auto t1 = omp_get_wtime();
	//auto v1 = average(buf.get(), N);
	//auto t2 = omp_get_wtime();
	//std::cout << "Result (classical): " << v1 << "\n";
	//std::cout << "Time taken (classical): " << t2 - t1 << "\n";
	//auto t3 = omp_get_wtime();
	//auto v2 = average_reduce(buf.get(), N);
	//auto t4 = omp_get_wtime();
	//std::cout << "Result (reduce): " << v2 << "\n";
	//std::cout << "Time taken (reduce): " << t4 - t3 << "\n";
	//auto t5 = omp_get_wtime();
	//auto v3 = average_rr(buf.get(), N);
	//auto t6 = omp_get_wtime();
	//std::cout << "Result (Round Robin): " << v3 << "\n";
	//std::cout << "Time taken (Round Robin): " << t6 - t5 << "\n";
	//auto t7 = omp_get_wtime();
	//auto v4 = average_omp(buf.get(), N);
	//auto t8 = omp_get_wtime();
	//std::cout << "Result (Open MP): " << v4 << "\n";
	//std::cout << "Time taken (Open MP): " << t8 - t7 << "\n";
	//auto t9 = omp_get_wtime();
	//auto v5 = average_omp_modified(buf.get(), N);
	//auto t10 = omp_get_wtime();
	//std::cout << "Result (Open MP modified): " << v5 << "\n";
	//std::cout << "Time taken (Open MP modified): " << t10 - t9 << "\n";
	//auto t11 = omp_get_wtime();
	//auto v6 = average_omp_mtx(buf.get(), N);
	//auto t12 = omp_get_wtime();
	//std::cout << "Result (Open MP Mutex): " << v6 << "\n";
	//std::cout << "Time taken (Open MP Mutex): " << t12 - t11 << "\n";
	//auto t13 = omp_get_wtime();
	//auto v7 = average_omp_mtx_modified(buf.get(), N);
	//auto t14 = omp_get_wtime();
	//std::cout << "Result (Open MP Mutex modified): " << v7 << "\n";
	//std::cout << "Time taken (Open MP Mutex modified): " << t14 - t13 << "\n";
	//auto t15 = omp_get_wtime();
	//auto v8 = average_cpp_mtx(buf.get(), N);
	//auto t16 = omp_get_wtime();
	//std::cout << "Result (CPP Mutex): " << v8 << "\n";
	//std::cout << "Time taken (CPP Mutex): " << t16 - t15 << "\n";
	//auto t17 = omp_get_wtime();
	//auto v9 = average_cpp_mtx_modified1(buf.get(), N);
	//auto t18 = omp_get_wtime();
	//std::cout << "Result (CPP Mutex modified0): " << v9 << "\n";
	//std::cout << "Time taken (CPP Mutex modified0): " << t18 - t17 << "\n";
	//auto t19 = omp_get_wtime();
	//auto v10 = average_cpp_mtx_local(buf.get(), N);
	//auto t20 = omp_get_wtime();
	//std::cout << "Result (CPP Mutex local): " << v10 << "\n";
	//std::cout << "Time taken (CPP Mutex local): " << t20 - t19 << "\n";
	//auto t21 = omp_get_wtime();
	//auto v11 = average_reduction_local(buf.get(), N);
	//auto t22 = omp_get_wtime();
	//std::cout << "Result (barrier): " << v11 << "\n";
	//std::cout << "Time taken (barrier): " << t22 - t21 << "\n";

	/*auto res = run_cpp_experiment(average_reduction_local, buf.get(), N);
	for (auto& r : res) {
		std::cout << r.res << "\t" << r.time << "\t\t" << r.speedup << "\t\t\t" << r.efficiency << std::endl;
	};*/

	/*unsigned p = std::thread::hardware_concurrency();
	auto producers = p / 2, consumers = p - producers;
	std::condition_variable cv;
	std::mutex mtx;
	std::queue<int> q;
	std::vector<std::thread> producers_v, consumers_v;
	for (auto i = 0; i < producers; ++i) {
		producers_v.emplace_back([&q, &mtx, &cv, consumers]() {
			for (auto j = 0; j < consumers; ++j) {
				std::scoped_lock lock(mtx);
				q.push(j);
			};
			cv.notify_all();
			});
	};
	for (auto i = 0; i < consumers; ++i) {
		consumers_v.emplace_back([&q, &mtx, &cv](unsigned t) {
			std::unique_lock ul(mtx);
			while (q.empty()) {
				cv.wait(ul);
			};
			int m = q.front();
			q.pop();
			std::cout << "Thread " << t << " received message " << m << "\n";
			}, i);
	};
	for (auto& producer : producers_v)
		producer.join();
	for (auto& consumer : consumers_v)
		consumer.join();*/
	//std::vector<uint32_t>v1(1000000);
	std::vector<std::pair<std::string, std::function<double(const double*, size_t)>>> funcs{
		{"reduce", average_rr},
		{"omp", average_omp},
		{"omp mod", average_omp_modified},
		{"omp mtx mod", average_omp_mtx_modified},
		{"cpp mtx", average_cpp_mtx},
		{"cpp mtx local", average_cpp_mtx_local},
		{"cpp barrier", average_reduction_local}
	};
	for (auto& func : funcs) {
		auto rr = run_cpp_experiment(func.second, buf.get(), N);
		if (_isatty(_fileno(stdout)))
			for (auto& r : rr)
				std::cout<<func.first<<"\t" << r.T << "\t" << r.res << "\t" << r.time << "\t\t" << r.speedup << "\t\t\t" << r.efficiency << "\n";
		else
			for (auto& r : rr)
				std::cout << func.first<<";"<< r.T << ";" << r.res << ";" << r.time << ";" << r.speedup << ";" << r.efficiency << "\r";
	}
	//auto rr = run_cpp_experiment(average_cpp_mtx, buf.get(), N);
	
	//auto rr = run_cpp_experiment_generate(randomize_vector_par, v1, 0, 0, 5);

	
	
	return 0;
}