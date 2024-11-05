// Copyright 2024 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/sidorina_p_check_lexicographic_order/include/ops_mpi.hpp"

TEST(sidorina_p_check_lexicographic_order_mpi, Test_difference_1st_element_0) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> str_ = {{'e', 'f', 'g', 'u'}, {'z', 'f', 'l', 'p'}};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataPar->inputs_count.emplace_back(str_.size());
    taskDataPar->inputs_count.emplace_back(str_[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  sidorina_p_check_lexicographic_order_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> ref_res(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataSeq->inputs_count.emplace_back(str_.size());
    taskDataSeq->inputs_count.emplace_back(str_[0].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_res.data()));
    taskDataSeq->outputs_count.emplace_back(ref_res.size());
    sidorina_p_check_lexicographic_order_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_res[0], res[0]);
    ASSERT_EQ(0, res[0]);
  }
}

TEST(sidorina_p_check_lexicographic_order_mpi, Test_difference_1st_element_1) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> str_ = {{'z', 'f', 'g', 'u'}, {'e', 'f', 'l', 'p'}};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataPar->inputs_count.emplace_back(str_.size());
    taskDataPar->inputs_count.emplace_back(str_[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  sidorina_p_check_lexicographic_order_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> ref_res(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataSeq->inputs_count.emplace_back(str_.size());
    taskDataSeq->inputs_count.emplace_back(str_[0].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_res.data()));
    taskDataSeq->outputs_count.emplace_back(ref_res.size());
    sidorina_p_check_lexicographic_order_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_res[0], res[0]);
    ASSERT_EQ(1, res[0]);
  }
}
TEST(sidorina_p_check_lexicographic_order_mpi, Test_difference_2nd_element_0) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> str_ = {{'e', 'f', 'g', 'u'}, {'e', 'x', 'l', 'p'}};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataPar->inputs_count.emplace_back(str_.size());
    taskDataPar->inputs_count.emplace_back(str_[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  sidorina_p_check_lexicographic_order_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> ref_res(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataSeq->inputs_count.emplace_back(str_.size());
    taskDataSeq->inputs_count.emplace_back(str_[0].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_res.data()));
    taskDataSeq->outputs_count.emplace_back(ref_res.size());
    sidorina_p_check_lexicographic_order_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_res[0], res[0]);
    ASSERT_EQ(0, res[0]);
  }
}
TEST(sidorina_p_check_lexicographic_order_mpi, Test_difference_2nd_element_1) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> str_ = {{'e', 'x', 'g', 'u'}, {'e', 'f', 'l', 'p'}};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataPar->inputs_count.emplace_back(str_.size());
    taskDataPar->inputs_count.emplace_back(str_[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  sidorina_p_check_lexicographic_order_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> ref_res(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataSeq->inputs_count.emplace_back(str_.size());
    taskDataSeq->inputs_count.emplace_back(str_[0].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_res.data()));
    taskDataSeq->outputs_count.emplace_back(ref_res.size());
    sidorina_p_check_lexicographic_order_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_res[0], res[0]);
    ASSERT_EQ(1, res[0]);
  }
}
TEST(sidorina_p_check_lexicographic_order_mpi, Test_difference_3d_element_0) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> str_ = {{'e', 'f', 'g', 'u'}, {'e', 'f', 'l', 'p'}};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataPar->inputs_count.emplace_back(str_.size());
    taskDataPar->inputs_count.emplace_back(str_[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  sidorina_p_check_lexicographic_order_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> ref_res(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataSeq->inputs_count.emplace_back(str_.size());
    taskDataSeq->inputs_count.emplace_back(str_[0].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_res.data()));
    taskDataSeq->outputs_count.emplace_back(ref_res.size());
    sidorina_p_check_lexicographic_order_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_res[0], res[0]);
    ASSERT_EQ(0, res[0]);
  }
}
TEST(sidorina_p_check_lexicographic_order_mpi, Test_difference_3d_element_1) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> str_ = {{'e', 'f', 'l', 'u'}, {'e', 'f', 'g', 'p'}};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataPar->inputs_count.emplace_back(str_.size());
    taskDataPar->inputs_count.emplace_back(str_[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  sidorina_p_check_lexicographic_order_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> ref_res(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataSeq->inputs_count.emplace_back(str_.size());
    taskDataSeq->inputs_count.emplace_back(str_[0].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_res.data()));
    taskDataSeq->outputs_count.emplace_back(ref_res.size());
    sidorina_p_check_lexicographic_order_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_res[0], res[0]);
    ASSERT_EQ(1, res[0]);
  }
}
TEST(sidorina_p_check_lexicographic_order_mpi, Test_difference_4_element_0) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> str_ = {{'e', 'f', 'g', 'a'}, {'e', 'f', 'g', 'p'}};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataPar->inputs_count.emplace_back(str_.size());
    taskDataPar->inputs_count.emplace_back(str_[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  sidorina_p_check_lexicographic_order_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> ref_res(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataSeq->inputs_count.emplace_back(str_.size());
    taskDataSeq->inputs_count.emplace_back(str_[0].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_res.data()));
    taskDataSeq->outputs_count.emplace_back(ref_res.size());
    sidorina_p_check_lexicographic_order_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_res[0], res[0]);
    ASSERT_EQ(0, res[0]);
  }
}
TEST(sidorina_p_check_lexicographic_order_mpi, Test_difference_4_element_1) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> str_ = {{'e', 'f', 'g', 'z'}, {'e', 'f', 'g', 'p'}};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataPar->inputs_count.emplace_back(str_.size());
    taskDataPar->inputs_count.emplace_back(str_[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  sidorina_p_check_lexicographic_order_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> ref_res(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataSeq->inputs_count.emplace_back(str_.size());
    taskDataSeq->inputs_count.emplace_back(str_[0].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_res.data()));
    taskDataSeq->outputs_count.emplace_back(ref_res.size());
    sidorina_p_check_lexicographic_order_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_res[0], res[0]);
    ASSERT_EQ(1, res[0]);
  }
}
TEST(sidorina_p_check_lexicographic_order_mpi, Test_6_and_5_elements) {
  boost::mpi::communicator world;
  std::vector<std::vector<char>> str_ = {{'e', 'f', 'g', 'k', 'l', 'm'}, {'e', 'f', 'g', 'k', 'l'}};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataPar->inputs_count.emplace_back(str_.size());
    taskDataPar->inputs_count.emplace_back(str_[0].size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  sidorina_p_check_lexicographic_order_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> ref_res(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[0].data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(str_[1].data()));
    taskDataSeq->inputs_count.emplace_back(str_.size());
    taskDataSeq->inputs_count.emplace_back(str_[0].size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ref_res.data()));
    taskDataSeq->outputs_count.emplace_back(ref_res.size());
    sidorina_p_check_lexicographic_order_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(ref_res[0], res[0]);
    ASSERT_EQ(1, res[0]);
  }
}