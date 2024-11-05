#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/voroshilov_v_num_of_alphabetic_chars/include/ops_mpi.hpp"

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

TEST(voroshilov_v_num_of_alphabetic_chars_mpi_func, test_without_alphabetic_chars_mpi) {
  std::string str = "123456789-+*/=<>";
  int initial_num = 0;
  int expected_num = 0;

  boost::mpi::communicator world;
  std::vector<char> global_vec(str.length());
  std::vector<int32_t> global_num(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::copy(str.begin(), str.end(), global_vec.begin());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_num.data()));
    taskDataPar->outputs_count.emplace_back(global_num.size());
  }

  voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskParallel alphabetCharsTaskParallel(taskDataPar);
  ASSERT_EQ(alphabetCharsTaskParallel.validation(), true);
  alphabetCharsTaskParallel.pre_processing();
  alphabetCharsTaskParallel.run();
  alphabetCharsTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Check if global_num is right
    ASSERT_EQ(expected_num, global_num[0]);

    // Create data
    std::vector<int32_t> reference_sum(1, initial_num);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskSequential alphabetCharsTaskSequential(taskDataSeq);
    ASSERT_EQ(alphabetCharsTaskSequential.validation(), true);
    alphabetCharsTaskSequential.pre_processing();
    alphabetCharsTaskSequential.run();
    alphabetCharsTaskSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_num[0]);
  }
}

TEST(voroshilov_v_num_of_alphabetic_chars_mpi_func, test_with_lowercase_alphabetic_chars_mpi) {
  std::string str = "123456789-+*/=<>aaabbcxyyzzz";
  int initial_num = 0;
  int expected_num = 12;

  boost::mpi::communicator world;
  std::vector<char> global_vec(str.length());
  std::vector<int32_t> global_num(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::copy(str.begin(), str.end(), global_vec.begin());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_num.data()));
    taskDataPar->outputs_count.emplace_back(global_num.size());
  }

  voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskParallel alphabetCharsTaskParallel(taskDataPar);
  ASSERT_EQ(alphabetCharsTaskParallel.validation(), true);
  alphabetCharsTaskParallel.pre_processing();
  alphabetCharsTaskParallel.run();
  alphabetCharsTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Check if global_num is right
    ASSERT_EQ(expected_num, global_num[0]);

    // Create data
    std::vector<int32_t> reference_sum(1, initial_num);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskSequential alphabetCharsTaskSequential(taskDataSeq);
    ASSERT_EQ(alphabetCharsTaskSequential.validation(), true);
    alphabetCharsTaskSequential.pre_processing();
    alphabetCharsTaskSequential.run();
    alphabetCharsTaskSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_num[0]);
  }
}

TEST(voroshilov_v_num_of_alphabetic_chars_mpi_func, test_with_uppercase_alphabetic_chars_mpi) {
  std::string str = "123456789-+*/=<>AAABBCXYYZZZ";
  int initial_num = 0;
  int expected_num = 12;

  boost::mpi::communicator world;
  std::vector<char> global_vec(str.length());
  std::vector<int32_t> global_num(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::copy(str.begin(), str.end(), global_vec.begin());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_num.data()));
    taskDataPar->outputs_count.emplace_back(global_num.size());
  }

  voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskParallel alphabetCharsTaskParallel(taskDataPar);
  ASSERT_EQ(alphabetCharsTaskParallel.validation(), true);
  alphabetCharsTaskParallel.pre_processing();
  alphabetCharsTaskParallel.run();
  alphabetCharsTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Check if global_num is right
    ASSERT_EQ(expected_num, global_num[0]);

    // Create data
    std::vector<int32_t> reference_sum(1, initial_num);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskSequential alphabetCharsTaskSequential(taskDataSeq);
    ASSERT_EQ(alphabetCharsTaskSequential.validation(), true);
    alphabetCharsTaskSequential.pre_processing();
    alphabetCharsTaskSequential.run();
    alphabetCharsTaskSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_num[0]);
  }
}

TEST(voroshilov_v_num_of_alphabetic_chars_mpi_func, test_with_anycase_alphabetic_chars_mpi) {
  std::string str = "123456789-+*/=<>aaabbcxyyzzzAAABBCXYYZZZ";
  int initial_num = 0;
  int expected_num = 24;

  boost::mpi::communicator world;
  std::vector<char> global_vec(str.length());
  std::vector<int32_t> global_num(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::copy(str.begin(), str.end(), global_vec.begin());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_num.data()));
    taskDataPar->outputs_count.emplace_back(global_num.size());
  }

  voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskParallel alphabetCharsTaskParallel(taskDataPar);
  ASSERT_EQ(alphabetCharsTaskParallel.validation(), true);
  alphabetCharsTaskParallel.pre_processing();
  alphabetCharsTaskParallel.run();
  alphabetCharsTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Check if global_num is right
    ASSERT_EQ(expected_num, global_num[0]);

    // Create data
    std::vector<int32_t> reference_sum(1, initial_num);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskSequential alphabetCharsTaskSequential(taskDataSeq);
    ASSERT_EQ(alphabetCharsTaskSequential.validation(), true);
    alphabetCharsTaskSequential.pre_processing();
    alphabetCharsTaskSequential.run();
    alphabetCharsTaskSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_num[0]);
  }
}

TEST(voroshilov_v_num_of_alphabetic_chars_mpi_func, test_with_random_generated_vector_mpi) {
  int initial_num = 0;
  int expected_num = 50;
  size_t vec_size = 100;

  boost::mpi::communicator world;
  std::vector<char> global_vec(vec_size);
  std::vector<int32_t> global_num(1, initial_num);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = genVecWithFixedAlphabeticsCount(expected_num, vec_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_num.data()));
    taskDataPar->outputs_count.emplace_back(global_num.size());
  }

  voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskParallel alphabetCharsTaskParallel(taskDataPar);
  ASSERT_EQ(alphabetCharsTaskParallel.validation(), true);
  alphabetCharsTaskParallel.pre_processing();
  alphabetCharsTaskParallel.run();
  alphabetCharsTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Check if global_num is right
    ASSERT_EQ(expected_num, global_num[0]);

    // Create data
    std::vector<int32_t> reference_sum(1, initial_num);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_sum.data()));
    taskDataSeq->outputs_count.emplace_back(reference_sum.size());

    // Create Task
    voroshilov_v_num_of_alphabetic_chars_mpi::AlphabetCharsTaskSequential alphabetCharsTaskSequential(taskDataSeq);
    ASSERT_EQ(alphabetCharsTaskSequential.validation(), true);
    alphabetCharsTaskSequential.pre_processing();
    alphabetCharsTaskSequential.run();
    alphabetCharsTaskSequential.post_processing();

    ASSERT_EQ(reference_sum[0], global_num[0]);
  }
}