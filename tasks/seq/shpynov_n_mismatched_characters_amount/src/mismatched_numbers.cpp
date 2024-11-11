#include "seq/shpynov_n_mismatched_characters_amount/include/mismatched_numbers.hpp"

#include <thread>
#include <vector>

using namespace std::chrono_literals;

int unique_characters(std::vector<std::string> const &vec1) {  // count unique characters
  std::string s1 = vec1[0];
  std::string s2 = vec1[1];
  int diff = abs(int(s1.size() - s2.size()));  // difference of strings' lengths is automatically included in the answer
  int count = diff;
  for (unsigned int i = 0; i < std::min(s1.size(), s2.size()); i++) {
    if (s1[i] != s2[i]) count++;
  }
  return count;
}

bool shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // receiving input data
  input_.push_back(std::string(reinterpret_cast<char *>(taskData->inputs[0])));
  input_.push_back(std::string(reinterpret_cast<char *>(taskData->inputs[1])));

  result = 0;

  return true;
}

bool shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential::validation() {
  internal_order_test();
  return ((taskData->inputs_count[0] == 2) && (taskData->outputs_count[0] == 1));
}

bool shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential::run() {
  internal_order_test();
  result = unique_characters(input_);  // counting unique characters in received strings
  return true;
}

bool shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<int *>(taskData->outputs[0])[0] = result;  // returning result
  return true;
}
