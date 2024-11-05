#include "seq/laganina_e_sum_values_by_columns_matrix/include/ops_seq.hpp"

#include <thread>
#include <vector>


bool laganina_e_sum_values_by_columns_matrix_seq::sum_values_by_columns_matrix_Seq::pre_processing() {
	internal_order_test();
	input_ = std::vector<int>(taskData->inputs_count[0]);
	m = taskData->inputs_count[1];
	n = taskData->inputs_count[2];
	auto* ptr = reinterpret_cast<int*>(taskData->inputs[0]);
	for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
		input_[i] = ptr[i];
	}
	res_ = std::vector<int>(n, 0);
	return true;
}

bool laganina_e_sum_values_by_columns_matrix_seq::sum_values_by_columns_matrix_Seq::validation() {
	internal_order_test();
	if (taskData->inputs_count[2] != taskData->outputs_count[0]) {
		return false;
	};
	if (taskData->inputs_count[1] < 1 || taskData->inputs_count[2] < 1) {
		return false;
	}
	if (taskData->inputs_count[0] != taskData->inputs_count[1] * taskData->inputs_count[2]) {
		return false;
	}
	return true;
}

bool laganina_e_sum_values_by_columns_matrix_seq::sum_values_by_columns_matrix_Seq::run() {
	internal_order_test();
	for (int j = 0; j < n; j++) {
		int sum = 0;
		for (int i = 0; i < m; i++) {
			sum += input_[i * n + j];
		}
		res_[j] = sum;
	}
	return true;
}

bool laganina_e_sum_values_by_columns_matrix_seq::sum_values_by_columns_matrix_Seq::post_processing() {
	internal_order_test();
	for (int i = 0; i < n; i++) {
		reinterpret_cast<int*>(taskData->outputs[0])[i] = res_[i];
	}
	return true;
}