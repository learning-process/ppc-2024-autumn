#include "seq/solovev_a_word_count/include/ops_seq.hpp"

namespace solovev_a_word_count_seq {

std::string create_text(int quan_words) {
std::string res;
std::string word = "word ";
for (int i = 0; i < quan_words; i++){
res += word;
}
return res;
}

int word_count(const std::string& input) {
std::istringstream iss(input);
return std::distance(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>());
}

bool solovev_a_word_count_seq::TestTaskSequential::pre_processing() {
internal_order_test();
input_ = std::string(reinterpret_cast<char*>(taskData->inputs[0]), taskData->inputs_count[0]);
res = 0;
return true;
}

bool solovev_a_word_count_seq::TestTaskSequential::validation() {
internal_order_test();
return taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1;
}

bool solovev_a_word_count_seq::TestTaskSequential::run() {
internal_order_test();
res = word_count(input_);
return true;
}

bool solovev_a_word_count_seq::TestTaskSequential::post_processing() {
internal_order_test();
reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
return true;
}

}  // namespace solovev_a_word_count_seq