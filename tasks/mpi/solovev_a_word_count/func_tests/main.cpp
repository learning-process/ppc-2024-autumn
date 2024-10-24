#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/solovev_a_word_count/include/ops_mpi.hpp"

TEST(solovev_a_word_count_mpi, test_em_str) {
  std::vector<char> input;
  std::vector<int> global_out(1, 0);
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  solovev_a_word_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int> ref_out(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_out.data()));
    taskDataSeq->outputs_count.emplace_back(ref_out.size());

    solovev_a_word_count_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_out[0], global_out[0]);
  }
}

TEST(solovev_a_word_count_mpi, test_120_word) {
  std::vector<char> input = solovev_a_word_count_mpi::create_text(120);
  std::vector<int> global_out(1, 0);
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  solovev_a_word_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int> ref_out(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_out.data()));
    taskDataSeq->outputs_count.emplace_back(ref_out.size());

    solovev_a_word_count_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_out[0], global_out[0]);
  }
}

TEST(solovev_a_word_count_mpi, test_300_words) {
  std::vector<char> input = solovev_a_word_count_mpi::create_text(300);
  std::vector<int> global_out(1, 0);
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  solovev_a_word_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int> ref_out(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_out.data()));
    taskDataSeq->outputs_count.emplace_back(ref_out.size());

    solovev_a_word_count_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_out[0], global_out[0]);
  }
}

TEST(solovev_a_word_count_mpi, test_480_words) {
  std::vector<char> input = solovev_a_word_count_mpi::create_text(480);
  std::vector<int> global_out(1, 0);
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  solovev_a_word_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int> ref_out(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_out.data()));
    taskDataSeq->outputs_count.emplace_back(ref_out.size());

    solovev_a_word_count_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_out[0], global_out[0]);
  }
}

TEST(solovev_a_word_count_mpi, test_600_words) {
  std::vector<char> input = solovev_a_word_count_mpi::create_text(600);
  std::vector<int> global_out(1, 0);
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  solovev_a_word_count_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int> ref_out(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    taskDataSeq->inputs_count.emplace_back(input.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_out.data()));
    taskDataSeq->outputs_count.emplace_back(ref_out.size());

    solovev_a_word_count_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_out[0], global_out[0]);
  }
}

