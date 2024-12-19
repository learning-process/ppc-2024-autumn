#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/map.hpp>
#include <random>
#include <vector>

#include "mpi/alputov_i_most_different_neighbor_elements/include/ops_mpi.hpp"

namespace alputov_i_most_different_neighbor_elements_mpi {
std::vector<int> generator(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());

  std::vector<int> ans(sz);
  for (int i = 0; i < sz; ++i) {
    ans[i] = gen() % 1000;
    int x = gen() % 2;
    if (x == 0) ans[i] *= -1;
  }

  return ans;
}
}  // namespace alputov_i_most_different_neighbor_elements_mpi

TEST(alputov_i_most_different_neighbor_elements_mpi, EmptyInput_ReturnsFalse) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(1);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::vector<int> reference_ans(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_ans.data()));
    taskDataSeq->outputs_count.emplace_back(reference_ans.size());

    alputov_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq
        most_different_neighbor_elements_seq(taskDataSeq);
    ASSERT_EQ(most_different_neighbor_elements_seq.validation(), false);
  }
}

TEST(alputov_i_most_different_neighbor_elements_mpi, InputSizeTwo_CorrectResult) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_diff(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int sz = 2;
    global_vec = std::vector<int>(sz, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_diff.data()));
    taskDataPar->outputs_count.emplace_back(global_diff.size());
  }

  alputov_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_diff(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_diff.data()));
    taskDataSeq->outputs_count.emplace_back(reference_diff.size());

    alputov_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq
        most_different_neighbor_elements_seq(taskDataSeq);
    ASSERT_EQ(most_different_neighbor_elements_seq.validation(), true);
    most_different_neighbor_elements_seq.pre_processing();
    most_different_neighbor_elements_seq.run();
    most_different_neighbor_elements_seq.post_processing();

    ASSERT_EQ(reference_diff[0], global_diff[0]);
  }
}
TEST(alputov_i_most_different_neighbor_elements_mpi, LargeRandomInput_CorrectResult) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int sz = 1234;
    global_vec = alputov_i_most_different_neighbor_elements_mpi::generator(sz);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  alputov_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    alputov_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq
        most_different_neighbor_elements_seq(taskDataSeq);
    ASSERT_EQ(most_different_neighbor_elements_seq.validation(), true);
    most_different_neighbor_elements_seq.pre_processing();
    most_different_neighbor_elements_seq.run();
    most_different_neighbor_elements_seq.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(alputov_i_most_different_neighbor_elements_mpi, MediumRandomInput_CorrectResult) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int sz = 120;
    global_vec = alputov_i_most_different_neighbor_elements_mpi::generator(sz);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  alputov_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    alputov_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq
        most_different_neighbor_elements_seq(taskDataSeq);
    ASSERT_EQ(most_different_neighbor_elements_seq.validation(), true);
    most_different_neighbor_elements_seq.pre_processing();
    most_different_neighbor_elements_seq.run();
    most_different_neighbor_elements_seq.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(alputov_i_most_different_neighbor_elements_mpi, AllEqualElements_CorrectResult) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int sz = 100;
    global_vec = std::vector<int>(sz, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  alputov_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    alputov_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq
        most_different_neighbor_elements_seq(taskDataSeq);
    ASSERT_EQ(most_different_neighbor_elements_seq.validation(), true);
    most_different_neighbor_elements_seq.pre_processing();
    most_different_neighbor_elements_seq.run();
    most_different_neighbor_elements_seq.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(alputov_i_most_different_neighbor_elements_mpi, AlternatingElements_CorrectResult) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {1, -1, 1, -1, 1, -1, 1, -1, 1, -1};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  alputov_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    alputov_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq
        most_different_neighbor_elements_seq(taskDataSeq);
    ASSERT_EQ(most_different_neighbor_elements_seq.validation(), true);
    most_different_neighbor_elements_seq.pre_processing();
    most_different_neighbor_elements_seq.run();
    most_different_neighbor_elements_seq.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(alputov_i_most_different_neighbor_elements_mpi, ConstantDifferenceSequence_CorrectResult) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int sz = 123;
    global_vec.resize(sz);
    for (int i = 0; i < sz; ++i) {
      global_vec[i] = sz - i;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  alputov_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    alputov_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq
        most_different_neighbor_elements_seq(taskDataSeq);
    ASSERT_EQ(most_different_neighbor_elements_seq.validation(), true);
    most_different_neighbor_elements_seq.pre_processing();
    most_different_neighbor_elements_seq.run();
    most_different_neighbor_elements_seq.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}

TEST(alputov_i_most_different_neighbor_elements_mpi, MostlyZerosInput_ReturnsCorrectPair) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_max(1);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = {12, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  alputov_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_mpi testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_max(1);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    alputov_i_most_different_neighbor_elements_mpi::most_different_neighbor_elements_seq
        most_different_neighbor_elements_seq(taskDataSeq);
    ASSERT_EQ(most_different_neighbor_elements_seq.validation(), true);
    most_different_neighbor_elements_seq.pre_processing();
    most_different_neighbor_elements_seq.run();
    most_different_neighbor_elements_seq.post_processing();

    ASSERT_EQ(reference_max[0], global_max[0]);
  }
}