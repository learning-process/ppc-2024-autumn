#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>

#include "mpi/deryabin_m_symbol_frequency/include/ops_mpi.hpp"

TEST(deryabin_m_symbol_frequency_mpi, test_shuffle) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<char> input_symbol(1, 'A');
  std::vector<int32_t> global_frequency(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_str = {'A', '1', 'A', '3', 'A', '5', 'A', '7', 'A', '9', 'A', 'B', 'A', 'D', 'A', 'F',
                  'A', 'H', 'A', 'J', 'A', 'L', 'A', 'N', 'A', 'P', 'A', 'R', 'A', 'T', 'A', 'V',
                  'A', 'X', 'A', 'Z', 'A', 'b', 'A', 'd', 'A', 'f', 'A', 'h', 'A', 'j', 'A', 'l',
                  'A', 'n', 'A', 'p', 'A', 'r', 'A', 't', 'A', 'v', 'A', 'x', 'A', 'z'};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(global_str.begin(), global_str.end(), gen);
    std::uniform_int_distribution<> distrib(1, 62);
    auto first = global_str.begin() + distrib(gen);
    auto last = global_str.end();
    global_str.erase(first, last);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataPar->inputs_count.emplace_back(input_symbol.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_frequency.data()));
    taskDataPar->outputs_count.emplace_back(global_frequency.size());
  }

  deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_frequency(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataSeq->inputs_count.emplace_back(input_symbol.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_frequency.data()));
    taskDataSeq->outputs_count.emplace_back(reference_frequency.size());

    // Create Task
    deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_frequency[0], global_frequency[0]);
  }
}

TEST(deryabin_m_symbol_frequency_mpi, test_empty) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<char> input_symbol(1, 'A');
  std::vector<int32_t> global_frequency(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataPar->inputs_count.emplace_back(input_symbol.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_frequency.data()));
    taskDataPar->outputs_count.emplace_back(global_frequency.size());
  }

  deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  ASSERT_EQ(0, global_frequency[0]);
}

TEST(deryabin_m_symbol_frequency_mpi, test_first_last) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<char> input_symbol(1, 'A');
  std::vector<int32_t> global_frequency(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_str = std::vector<char>(1000, 'B');
    global_str[0] = global_str[999] = 'A';
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataPar->inputs_count.emplace_back(input_symbol.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_frequency.data()));
    taskDataPar->outputs_count.emplace_back(global_frequency.size());
  }

  deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_frequency(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataSeq->inputs_count.emplace_back(input_symbol.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_frequency.data()));
    taskDataSeq->outputs_count.emplace_back(reference_frequency.size());

    // Create Task
    deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_frequency[0], global_frequency[0]);
  }
}

TEST(deryabin_m_symbol_frequency_mpi, test_same_letters) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<char> input_symbol(1, 'A');
  std::vector<int32_t> global_frequency(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_str = std::vector<char>(1000, 'A');
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataPar->inputs_count.emplace_back(input_symbol.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_frequency.data()));
    taskDataPar->outputs_count.emplace_back(global_frequency.size());
  }

  deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_frequency(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataSeq->inputs_count.emplace_back(input_symbol.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_frequency.data()));
    taskDataSeq->outputs_count.emplace_back(reference_frequency.size());

    // Create Task
    deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_frequency[0], global_frequency[0]);
  }
}

TEST(deryabin_m_symbol_frequency_mpi, test_random) {
  boost::mpi::communicator world;
  std::vector<char> global_str;
  std::vector<char> input_symbol(1, 'A');
  std::vector<int32_t> global_frequency(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_str = std::vector<char>(10000);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(65, 90);
    std::generate(global_str.begin(), global_str.end(), [&] { return char(0) + distrib(gen); });
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataPar->inputs_count.emplace_back(global_str.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataPar->inputs_count.emplace_back(input_symbol.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_frequency.data()));
    taskDataPar->outputs_count.emplace_back(global_frequency.size());
  }

  deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_frequency(1, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_str.data()));
    taskDataSeq->inputs_count.emplace_back(global_str.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_symbol.data()));
    taskDataSeq->inputs_count.emplace_back(input_symbol.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_frequency.data()));
    taskDataSeq->outputs_count.emplace_back(reference_frequency.size());

    // Create Task
    deryabin_m_symbol_frequency_mpi::SymbolFrequencyMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_frequency[0], global_frequency[0]);
  }
}
