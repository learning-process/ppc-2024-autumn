#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

#include "seq/voroshilov_v_num_of_alphabetic_chars/include/ops_seq.hpp"

std::vector<char> genVecWithFixedAlphabeticsCount(int alphCount, size_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<char> vector(size);
  int curCount = 0;

  std::string charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&(){}[]*+-/";
  int charset_alphabet_size = 52;

  // Generate with absolutely random alphabetics count:
  for (size_t i = 0; i < vector.size(); i++) {
    int number = gen() % charset.length();
    vector[i] = charset[number];
    if (std::isalpha(vector[i]) != 0) {
      curCount++;
    }
  }

  if (curCount < alphCount) {
    // Change non-alphabetics to alphabetics to complete missing quantity
    for (size_t i = 0; curCount < alphCount; i++) {
      if (std::isalpha(vector[i]) == 0) {
        int number = gen() % charset_alphabet_size;
        vector[i] = charset[number];
        curCount++;
      }
    }
  } else {
    // Change alphabetics to non-alphabetics if there is an oversupply
    for (size_t i = 0; curCount > alphCount; i++) {
      if (std::isalpha(vector[i]) != 0) {
        int number = gen() % (charset.length() - charset_alphabet_size) + charset_alphabet_size;
        vector[i] = charset[number];
        curCount--;
      }
    }
  }

  return vector;
}

TEST(voroshilov_v_num_of_alphabetic_chars_seq_func, test_without_alphabetic_chars_seq) {
  std::string str = "123456789-+*/=<>";
  int initial_num = 0;
  int expected_num = 0;

  // Create data
  std::vector<char> in(str.length());
  std::copy(str.begin(), str.end(), in.begin());
  std::vector<int> out(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential alphabetCharsTaskSequential(taskDataSeq);
  ASSERT_EQ(alphabetCharsTaskSequential.validation(), true);
  alphabetCharsTaskSequential.pre_processing();
  alphabetCharsTaskSequential.run();
  alphabetCharsTaskSequential.post_processing();
  ASSERT_EQ(expected_num, out[0]);
}

TEST(voroshilov_v_num_of_alphabetic_chars_seq_func, test_with_lowercase_alphabetic_chars_seq) {
  std::string str = "123456789-+*/=<>aaabbcxyyzzz";
  int initial_num = 0;
  int expected_num = 12;

  // Create data
  std::vector<char> in(str.length());
  std::copy(str.begin(), str.end(), in.begin());
  std::vector<int> out(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential alphabetCharsTaskSequential(taskDataSeq);
  ASSERT_EQ(alphabetCharsTaskSequential.validation(), true);
  alphabetCharsTaskSequential.pre_processing();
  alphabetCharsTaskSequential.run();
  alphabetCharsTaskSequential.post_processing();
  ASSERT_EQ(expected_num, out[0]);
}

TEST(voroshilov_v_num_of_alphabetic_chars_seq_func, test_with_uppercase_alphabetic_chars_seq) {
  std::string str = "123456789-+*/=<>AAABBCXYYZZZ";
  int initial_num = 0;
  int expected_num = 12;

  // Create data
  std::vector<char> in(str.length());
  std::copy(str.begin(), str.end(), in.begin());
  std::vector<int> out(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential alphabetCharsTaskSequential(taskDataSeq);
  ASSERT_EQ(alphabetCharsTaskSequential.validation(), true);
  alphabetCharsTaskSequential.pre_processing();
  alphabetCharsTaskSequential.run();
  alphabetCharsTaskSequential.post_processing();
  ASSERT_EQ(expected_num, out[0]);
}

TEST(voroshilov_v_num_of_alphabetic_chars_seq_func, test_with_anycase_alphabetic_chars_seq) {
  std::string str = "123456789-+*/=<>aaabbcxyyzzzAAABBCXYYZZZ";
  int initial_num = 0;
  int expected_num = 24;

  // Create data
  std::vector<char> in(str.length());
  std::copy(str.begin(), str.end(), in.begin());
  std::vector<int> out(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential alphabetCharsTaskSequential(taskDataSeq);
  ASSERT_EQ(alphabetCharsTaskSequential.validation(), true);
  alphabetCharsTaskSequential.pre_processing();
  alphabetCharsTaskSequential.run();
  alphabetCharsTaskSequential.post_processing();
  ASSERT_EQ(expected_num, out[0]);
}

TEST(voroshilov_v_num_of_alphabetic_chars_seq_func, test_with_random_generated_vector_seq) {
  int initial_num = 0;
  int expected_num = 50;
  size_t vec_size = 100;

  // Create data
  std::vector<char> in = genVecWithFixedAlphabeticsCount(expected_num, vec_size);
  std::vector<int> out(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential alphabetCharsTaskSequential(taskDataSeq);
  ASSERT_EQ(alphabetCharsTaskSequential.validation(), true);
  alphabetCharsTaskSequential.pre_processing();
  alphabetCharsTaskSequential.run();
  alphabetCharsTaskSequential.post_processing();
  ASSERT_EQ(expected_num, out[0]);
}