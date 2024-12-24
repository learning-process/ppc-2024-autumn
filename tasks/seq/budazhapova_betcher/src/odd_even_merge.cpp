
#include "seq/budazhapova_odd_even_merge/include/odd_even_merge.hpp"
#include <algorithm>
#include <thread>

void budazhapova_betcher_odd_even_merge_seq::MergeSequential::counting_sort(std::vector<int>& arr, int exp) {
	int n = arr.size();
	std::vector<int> output(n);
	std::vector<int> count(10, 0);

	for (int i = 0; i < n; i++) {
		int index = (arr[i] / exp) % 10;
		count[index]++;
	}
	for (int i = 1; i < 10; i++) {
		count[i] += count[i - 1];
	}
	for (int i = n - 1; i >= 0; i--) {
		int index = (arr[i] / exp) % 10;
		output[count[index] - 1] = arr[i];
		count[index]--;
	}
	for (int i = 0; i < n; i++) {
		arr[i] = output[i];
	}
}

void budazhapova_betcher_odd_even_merge_seq::MergeSequential::radix_sort(std::vector<int>& arr) {
	int max_num = *std::max_element(arr.begin(), arr.end());
	for (int exp = 1; max_num / exp > 0; exp *= 10) {
		counting_sort(arr, exp);
	}
}

bool budazhapova_betcher_odd_even_merge_seq::MergeSequential::pre_processing() {
	internal_order_test();
	res = std::vector<int>(reinterpret_cast<int*>(taskData->inputs[0]),
		reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0]);
	return true;
}

bool budazhapova_betcher_odd_even_merge_seq::MergeSequential::validation() {
	internal_order_test();
	return taskData->inputs_count[1] > 0;
}

bool budazhapova_betcher_odd_even_merge_seq::MergeSequential::run() {
	internal_order_test();
	radix_sort(res);
	return true;
}

bool budazhapova_betcher_odd_even_merge_seq::MergeSequential::post_processing() {
	internal_order_test();
	int* output = reinterpret_cast<int*>(taskData->outputs[0]);
	for (int i = 0; i < rows; i++) {
		output[i] = res[i];
	}
	return true;
}
