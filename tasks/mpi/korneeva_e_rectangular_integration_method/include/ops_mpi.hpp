
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <vector>

#include "core/task/include/task.hpp"

namespace korneeva_e_rectangular_integration_method_mpi {

constexpr double MIN_EPSILON = 1e-6;
using Function = std::function<double(std::vector<double>& args)>;

class RectangularIntegrationSeq : public ppc::core::Task {
 public:
  explicit RectangularIntegrationSeq(std::shared_ptr<ppc::core::TaskData> taskData, Function& func)
      : Task(std::move(taskData)), integrandFunction(func) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double result;
  double epsilon;
  Function integrandFunction;
  std::vector<std::pair<double, double>> limits;
};

class RectangularIntegrationMPI : public ppc::core::Task {
 public:
  explicit RectangularIntegrationMPI(std::shared_ptr<ppc::core::TaskData> taskData, Function& func)
      : Task(std::move(taskData)), integrandFunction(func) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double result;
  double epsilon;
  Function integrandFunction;
  std::vector<std::pair<double, double>> limits;

  boost::mpi::communicator mpi_comm;  // MPI communicator
};

// Function for sequential integration (recursive)
double calculateIntegral(const Function& func_, double epsilon_, std::vector<std::pair<double, double>>& limits_,
                         std::vector<double>& args_);

// class RectangularIntegrationSeq
bool RectangularIntegrationSeq::pre_processing() {
  internal_order_test();

  auto* ptrInput = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);
  limits.assign(ptrInput, ptrInput + taskData->inputs_count[0]);
  result = 0.0;

  epsilon = *reinterpret_cast<double*>(taskData->inputs[1]);
  if (epsilon < MIN_EPSILON) {
    epsilon = MIN_EPSILON;
  }

  return true;
}

bool RectangularIntegrationSeq::validation() {
  internal_order_test();

  bool validInput = taskData->inputs_count[0] > 0 && taskData->inputs.size() == 2;
  bool validOutput = taskData->outputs_count[0] == 1 && !taskData->outputs.empty();

  size_t numDimensions = taskData->inputs_count[0];
  bool validLimits = true;
  auto* ptrInput = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);

  for (size_t i = 0; i < numDimensions; ++i) {
    if (ptrInput[i].first > ptrInput[i].second) {
      validLimits = false;
      break;
    }
  }
  return validInput && validOutput && validLimits;
}

bool RectangularIntegrationSeq::run() {
  internal_order_test();
  std::vector<double> args;
  result = calculateIntegral(integrandFunction, epsilon, limits, args);

  return true;
}

bool RectangularIntegrationSeq::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  return true;
}

// class RectangularIntegrationMPI
bool RectangularIntegrationMPI::pre_processing() {
  internal_order_test();

  if (mpi_comm.rank() == 0) {
    auto* ptr = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);
    limits.assign(ptr, ptr + taskData->inputs_count[0]);

    epsilon = *reinterpret_cast<double*>(taskData->inputs[1]);
    if (epsilon < MIN_EPSILON) {
      epsilon = MIN_EPSILON;
    }
  }
  result = 0.0;
  return true;
}

bool RectangularIntegrationMPI::validation() {
  internal_order_test();

  if (mpi_comm.rank() == 0) {
    bool validInput = (taskData->inputs_count[0] > 0 && taskData->inputs.size() == 2);
    bool validOutput = (taskData->outputs_count[0] == 1 && !taskData->outputs.empty());

    size_t numDimensions = taskData->inputs_count[0];
    bool validLimits = true;
    auto* ptrInput = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);

    for (size_t i = 0; i < numDimensions; ++i) {
      if (ptrInput[i].first > ptrInput[i].second) {
        validLimits = false;
        break;
      }
    }
    return validInput && validOutput && validLimits;
  }
  return true;
}

bool RectangularIntegrationMPI::run() {
  internal_order_test();

  std::vector<double> params;
  bool refining = true;

  broadcast(mpi_comm, limits, 0);
  broadcast(mpi_comm, epsilon, 0);

  int numProcs = mpi_comm.size();
  double globalSum = 0.0;
  double prevGlobalSum = 0.0;
  double localSum = 0.0;

  auto [start, end] = limits.front();
  double stepSize = (end - start) / numProcs;

  double localStart = start + stepSize * mpi_comm.rank();
  double localEnd = localStart + stepSize;

  limits.erase(limits.begin());
  params.emplace_back(0.0);

  while (refining) {
    prevGlobalSum = globalSum;
    globalSum = 0.0;
    localSum = 0.0;

    int localSegments = numProcs / mpi_comm.size();
    double segmentStep = (localEnd - localStart) / localSegments;
    params.back() = localStart + segmentStep / 2.0;

    for (int i = 0; i < localSegments; i++) {
      if (limits.empty()) {
        localSum += integrandFunction(params) * segmentStep;
      } else {
        localSum += calculateIntegral(integrandFunction, epsilon, limits, params) * segmentStep;
      }
      params.back() += segmentStep;
    }

    reduce(mpi_comm, localSum, globalSum, std::plus<>(), 0);

    if (mpi_comm.rank() == 0) {
      refining = (std::abs(globalSum - prevGlobalSum) * (1.0 / 3.0) > epsilon);
    }

    broadcast(mpi_comm, refining, 0);

    numProcs *= 2;
  }

  result = (mpi_comm.rank() == 0) ? globalSum : 0.0;
  return true;
}

bool RectangularIntegrationMPI::post_processing() {
  internal_order_test();
  if (mpi_comm.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  }
  return true;
}

// function calculateIntegral()
double calculateIntegral(const Function& func_, double epsilon_, std::vector<std::pair<double, double>>& limits_,
                         std::vector<double>& args_) {
  double integralValue = 0;
  double prevValue = 0;
  int subdivisions = 2;
  bool flag = true;

  auto [low, high] = limits_.front();
  limits_.erase(limits_.begin());
  args_.emplace_back(0.0);

  while (flag) {
    prevValue = integralValue;
    integralValue = 0.0;

    double step = (high - low) / subdivisions;
    args_.back() = low + step / 2.0;

    for (int i = 0; i < subdivisions; ++i) {
      if (limits_.empty()) {
        integralValue += func_(args_) * step;
      } else {
        integralValue += calculateIntegral(func_, epsilon_, limits_, args_) * step;
      }
      args_.back() += step;
    }

    subdivisions *= 2;
    flag = (std::abs(integralValue - prevValue) * (1.0 / 3.0) > epsilon_);
  }

  args_.pop_back();
  limits_.insert(limits_.begin(), {low, high});

  return integralValue;
}

}  // namespace korneeva_e_rectangular_integration_method_mpi