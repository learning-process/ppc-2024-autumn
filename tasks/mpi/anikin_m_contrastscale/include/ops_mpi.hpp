// Copyright 2024 Anikin Maksim
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace anikin_m_contrastscale_mpi {

struct RGB {
  uint8_t R = 0;
  uint8_t G = 0;
  uint8_t B = 0;

  // clang-format off
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & R;
    ar & G;
    ar & B;
  }
  // clang-format on
};

RGB getrandomRGB();
double getcontrast(std::vector<RGB>& in);

class ContrastScaleSeq : public ppc::core::Task {
 public:
  explicit ContrastScaleSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  float correction;
  int iab;
  std::vector<RGB> input_, output_;
};

class ContrastScaleMpi : public ppc::core::Task {
 public:
  explicit ContrastScaleMpi(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  float correction;
  std::vector<RGB> input_, output_;
  boost::mpi::communicator world;
};

}  // namespace anikin_m_contrastscale_mpi