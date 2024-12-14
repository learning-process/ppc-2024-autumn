#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
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

TEST(voroshilov_v_num_of_alphabetic_chars_seq_perf, test_pipeline_run_seq) {
  int initial_num = 0;
  int expected_num = 5000;
  size_t vec_size = 10000;

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
  auto alphabetCharsTaskSequential =
      std::make_shared<voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(alphabetCharsTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(expected_num, out[0]);
}

TEST(voroshilov_v_num_of_alphabetic_chars_seq_perf, test_task_run_seq) {
  int initial_num = 0;
  int expected_num = 5000;
  size_t vec_size = 10000;

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
  auto alphabetCharsTaskSequential =
      std::make_shared<voroshilov_v_num_of_alphabetic_chars_seq::AlphabetCharsTaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(alphabetCharsTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(expected_num, out[0]);
}
