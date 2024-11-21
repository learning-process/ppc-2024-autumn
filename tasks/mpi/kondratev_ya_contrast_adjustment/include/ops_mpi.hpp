// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace kondratev_ya_contrast_adjustment_mpi {

struct Pixel {
  uint8_t red;
  uint8_t green;
  uint8_t blue;

  Pixel& operator=(uint8_t value) {
    this->red = value;
    this->green = value;
    this->blue = value;

    return *this;
  }

  template <typename Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar& red;
    ar& green;
    ar& blue;
  }
};

int clamp(int value, int min, int max);
double getContrast(std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel>& array);

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input_, res_;
  double contrast_;
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<kondratev_ya_contrast_adjustment_mpi::Pixel> input_, res_;
  double contrast_;
  boost::mpi::communicator world;
};

}  // namespace kondratev_ya_contrast_adjustment_mpi
