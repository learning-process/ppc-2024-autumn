#include "seq/kudryashova_i_vector_dot_product/include/vectorDotProductSeq.hpp"
#include <random>
#include <thread>

using namespace std::chrono_literals;
static int seedOffset = 0;


std::vector<int> kudryashova_i_vector_dot_product::GetRandomVector(const int size) {
	std::vector<int> vector(size);
	std::srand(static_cast<unsigned>(time(NULL)) + ++seedOffset);
	for (int i = 0; i < size; ++i) {
		vector[i] = std::rand() % 10 + 1;
	}
	return vector;
}

int kudryashova_i_vector_dot_product::vectorDotProduct(const std::vector<int>& vector1, const std::vector<int>& vector2) {
	long long result = 0;
	for (unsigned long i = 0; i < vector1.size(); ++i) {
		result += vector1[i] * vector2[i];
	}
	return result;
}

bool kudryashova_i_vector_dot_product::TestTaskSequential::pre_processing() {
	internal_order_test();

	input_.resize(taskData->inputs.size());
	for (unsigned long i = 0; i < input_.size(); ++i) {
		auto* tempPtr = reinterpret_cast<int*>(taskData->inputs[i]);
		input_[i] = std::vector<int>(taskData->inputs_count[i]);
		std::copy(tempPtr, tempPtr + taskData->inputs_count[i], input_[i].begin());
	}
	result = 0;
	return true;
}


bool kudryashova_i_vector_dot_product::TestTaskSequential::validation() {
	internal_order_test();
	return (taskData->inputs_count[0] == taskData->inputs_count[1]) && 
		(taskData->inputs.size() == taskData->inputs_count.size() && taskData->inputs.size() == 2) &&
		taskData->outputs_count[0] == 1 &&
		(taskData->outputs.size() == taskData->outputs_count.size()) && taskData->outputs.size() == 1;
}

bool kudryashova_i_vector_dot_product::TestTaskSequential::run() {
	internal_order_test();
	for (unsigned long i = 0; i < input_[0].size(); i++) {
		result += input_[1][i] * input_[0][i];
	}
	return true;
}

 
bool kudryashova_i_vector_dot_product::TestTaskSequential::post_processing() {
	internal_order_test();
	reinterpret_cast<int*>(taskData->outputs[0])[0] = result;
	return true;
}
