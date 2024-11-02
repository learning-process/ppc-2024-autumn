#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstdint>
#include <random>
#include <vector>

#include "../include/ops_mpi.hpp"

void run_test(std::string &&in) {
  boost::mpi::communicator world;

  size_t out = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
    taskDataPar->outputs_count.emplace_back(1);
  }

  vedernikova_k_word_num_in_str_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    size_t ref = 0;

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>(*taskDataPar);
    taskDataSeq->outputs[0] = reinterpret_cast<uint8_t *>(&ref);

    vedernikova_k_word_num_in_str_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    EXPECT_EQ(out, ref);
  }
}

std::string make_random_sentence(size_t length) {
  std::string buf(length, ' ');

  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<> dist(0., 1.);

  for (size_t i = 0; i < length; i++) {
    if (dist(gen) > 0.6) {
      buf[i] = 'a';
    }
  }

  return buf;
}

TEST(vedernikova_k_word_num_in_str_mpi, empty) { run_test(""); }

TEST(vedernikova_k_word_num_in_str_mpi, empty_strlen_1) { run_test(std::string(1, ' ')); }
TEST(vedernikova_k_word_num_in_str_mpi, empty_strlen_2) { run_test(std::string(2, ' ')); }
TEST(vedernikova_k_word_num_in_str_mpi, empty_strlen_3) { run_test(std::string(3, ' ')); }
TEST(vedernikova_k_word_num_in_str_mpi, empty_strlen_4) { run_test(std::string(4, ' ')); }
TEST(vedernikova_k_word_num_in_str_mpi, empty_strlen_5) { run_test(std::string(5, ' ')); }

TEST(vedernikova_k_word_num_in_str_mpi, strlen_1) { run_test("1"); }
TEST(vedernikova_k_word_num_in_str_mpi, strlen_2) { run_test("2"); }
TEST(vedernikova_k_word_num_in_str_mpi, strlen_3) { run_test("3"); }
TEST(vedernikova_k_word_num_in_str_mpi, strlen_4) { run_test("4"); }

TEST(vedernikova_k_word_num_in_str_mpi, words_1) { run_test("Hello"); }
TEST(vedernikova_k_word_num_in_str_mpi, words_1_leading) { run_test(" Hello"); }
TEST(vedernikova_k_word_num_in_str_mpi, words_1_trailing) { run_test("Hello "); }
TEST(vedernikova_k_word_num_in_str_mpi, words_1_padded) { run_test(" Hello "); }

TEST(vedernikova_k_word_num_in_str_mpi, words_2) { run_test("Hello World"); }
TEST(vedernikova_k_word_num_in_str_mpi, words_2_leading) { run_test(" Hello World"); }
TEST(vedernikova_k_word_num_in_str_mpi, words_2_trailing) { run_test("Hello World "); }
TEST(vedernikova_k_word_num_in_str_mpi, words_2_padded) { run_test(" Hello World "); }
TEST(vedernikova_k_word_num_in_str_mpi, words_2_inner) { run_test("Hello  World"); }

TEST(vedernikova_k_word_num_in_str_mpi, words_3) { run_test("1 2 3"); }

TEST(vedernikova_k_word_num_in_str_seq, padded_corner) { run_test("  a  "); }

TEST(vedernikova_k_word_num_in_str_mpi, random_3) { run_test(make_random_sentence(3)); }
TEST(vedernikova_k_word_num_in_str_mpi, random_4) { run_test(make_random_sentence(4)); }
TEST(vedernikova_k_word_num_in_str_mpi, random_5) { run_test(make_random_sentence(5)); }

TEST(vedernikova_k_word_num_in_str_mpi, random_64) { run_test(make_random_sentence(64)); }
TEST(vedernikova_k_word_num_in_str_mpi, random_128) { run_test(make_random_sentence(128)); }
TEST(vedernikova_k_word_num_in_str_mpi, random_256) { run_test(make_random_sentence(256)); }
TEST(vedernikova_k_word_num_in_str_mpi, random_512) { run_test(make_random_sentence(512)); }
TEST(vedernikova_k_word_num_in_str_mpi, random_1024) { run_test(make_random_sentence(512)); }