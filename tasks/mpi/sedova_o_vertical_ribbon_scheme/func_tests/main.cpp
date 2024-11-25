// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/sedova_o_vertical_ribbon_scheme/include/ops_mpi.hpp"

#include <iostream>
#include <algorithm>
#include <random>
#include <boost/mpi.hpp>

std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(0, 1000);
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

std::vector<int> getRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> dist(0, 1000);
  std::vector<int> matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = dist(gen);
  }
  return matrix;
}
TEST(sedova_o_vertical_ribbon_scheme, FailValid) {
  std::vector<int> input_data = {46};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataSeq->inputs_count.emplace_back(1);

  sedova_o_vertical_ribbon_scheme_mpi::SequentialMPI taskSequential(taskDataSeq);
  ASSERT_EQ(taskSequential.validation(), false);
}