#include <gtest/gtest.h>

#include <vector>

#include "../include/ops_seq.hpp"

void run_test(std::string &&in, size_t solution) {
  size_t out = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  vedernikova_k_word_num_in_str_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_EQ(solution, out);
}

TEST(vedernikova_k_word_num_in_str_seq, empty) { run_test("", 0); }
TEST(vedernikova_k_word_num_in_str_seq, empty_strlen_1) { run_test(std::string(1, ' '), 0); }
TEST(vedernikova_k_word_num_in_str_seq, empty_strlen_2) { run_test(std::string(2, ' '), 0); }
TEST(vedernikova_k_word_num_in_str_seq, empty_strlen_3) { run_test(std::string(3, ' '), 0); }
TEST(vedernikova_k_word_num_in_str_seq, empty_strlen_4) { run_test(std::string(4, ' '), 0); }
TEST(vedernikova_k_word_num_in_str_seq, empty_strlen_5) { run_test(std::string(5, ' '), 0); }

TEST(vedernikova_k_word_num_in_str_seq, words_1) { run_test("Hello", 1); }
TEST(vedernikova_k_word_num_in_str_seq, words_1_leading) { run_test(" Hello", 1); }
TEST(vedernikova_k_word_num_in_str_seq, words_1_trailing) { run_test("Hello ", 1); }
TEST(vedernikova_k_word_num_in_str_seq, words_1_padded) { run_test(" Hello ", 1); }

TEST(vedernikova_k_word_num_in_str_seq, words_2) { run_test("Hello World", 2); }
TEST(vedernikova_k_word_num_in_str_seq, words_2_leading) { run_test(" Hello World", 2); }
TEST(vedernikova_k_word_num_in_str_seq, words_2_trailing) { run_test("Hello World ", 2); }
TEST(vedernikova_k_word_num_in_str_seq, words_2_padded) { run_test(" Hello World ", 2); }
TEST(vedernikova_k_word_num_in_str_seq, words_2_inner) { run_test("Hello  World", 2); }

TEST(vedernikova_k_word_num_in_str_seq, words_3) { run_test("1 2 3", 3); }
