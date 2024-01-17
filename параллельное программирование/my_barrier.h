#pragma once
#include <mutex>

class latch {
	unsigned T;
	std::mutex mtx;
	std::condition_variable cv;
public:
	latch(unsigned threads): T(threads){}
	void arrive_and_wait() {
		std::unique_lock l{ mtx };
		if (--T) do {
			cv.wait(l);
		} while (T > 0);
		else
			cv.notify_all();
	}
};

class barrier {
	unsigned lock_id, T, Tmax;
	std::mutex mtx;
	std::condition_variable cv;
public:
	barrier(unsigned threads) : lock_id(0), T(threads), Tmax(threads) {}
	void arrive_and_wait() {
		std::unique_lock l{ mtx };
		if (--T) {
			unsigned my_lock_id = lock_id;
			while (my_lock_id == lock_id) {
				cv.wait(l);
			}
		}
		else {
			++lock_id;
			cv.notify_all();
			T = Tmax;
		}
	}
};