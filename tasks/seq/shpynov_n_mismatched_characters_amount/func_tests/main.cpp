#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "seq/shpynov_n_mismatched_characters_amount/include/mismatched_numbers.hpp"

TEST(shpynov_n_amount_of_mismatched_numbers_seq, empty_strings) {
  // Create data

  std::string str1, str2 = "";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();  // Create TaskData
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential testTaskSequential(taskDataSeq);  // Create Task

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(0, out[0]);
}

TEST(shpynov_n_amount_of_mismatched_numbers_seq, only_one_string_is_empty) {
  std::string str1 = "";
  std::string str2 = "abcd";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ((int)str2.length(), out[0]);
}

TEST(shpynov_n_amount_of_mismatched_numbers_seq, throw_on_different_amount_of_strings) {
  std::string str1 = "";
  std::vector<std::string> v1{str1};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(shpynov_n_amount_of_mismatched_numbers_seq, same_strings) {
  std::string str1 = "abcd";
  std::string str2 = "abcd";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(0, out[0]);
}

TEST(shpynov_n_amount_of_mismatched_numbers_seq, correct_maths_3) {
  std::string str1 = "abcdff";
  std::string str2 = "abcevv";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(3, out[0]);
}

TEST(shpynov_n_amount_of_mismatched_numbers_seq, correct_maths_3_with_different_lengths) {
  std::string str1 = "abcd";
  std::string str2 = "abcevv";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(3, out[0]);
}

TEST(shpynov_n_amount_of_mismatched_numbers_seq, correct_maths_7) {
  std::string str1 = "aebdfcuvgdh";
  std::string str2 = "acbdbcedhug";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(7, out[0]);
}

TEST(shpynov_n_amount_of_mismatched_numbers_seq, correct_maths_7_with_different_lengths) {
  std::string str1 = "aebdfcuvgdhwf";
  std::string str2 = "acbdbcedhdh";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(7, out[0]);
}

TEST(shpynov_n_amount_of_mismatched_numbers_seq, correct_maths_11) {
  std::string str1 = "aebdfc@vgddddier/v";
  std::string str2 = "aedvfc@vnjgbj!eaew";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(11, out[0]);
}

TEST(shpynov_n_amount_of_mismatched_numbers_seq, correct_maths_11_with_different_lengths) {
  std::string str1 = "aebdfc@vgddddier/v";
  std::string str2 = "aeddfc@vndgbj!eaewd4";
  std::vector<std::string> v1{str1, str2};

  std::vector<int> out(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shpynov_n_amount_of_mismatched_numbers_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(11, out[0]);
}
