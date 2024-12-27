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

  boost::mpi::communicator mpi_comm;
};

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

  std::vector<double> args;

  broadcast(mpi_comm, limits, 0);
  broadcast(mpi_comm, epsilon, 0);

  double globalIntegral = 0.0;
  double previousGlobalIntegral = 0.0;
  double localIntegral = 0.0;
  int totalProcesses = mpi_comm.size();

  auto [intervalStart, intervalEnd] = limits.front();
  double intervalStep = (intervalEnd - intervalStart) / totalProcesses;

  double localStart = intervalStart + intervalStep * mpi_comm.rank();
  double localEnd = localStart + intervalStep;

  limits.erase(limits.begin());
  args.emplace_back(0.0);

  bool continueRefining = true;
  while (continueRefining) {
    previousGlobalIntegral = globalIntegral;
    globalIntegral = 0.0;
    localIntegral = 0.0;

    int localSegments = totalProcesses / mpi_comm.size();
    double segmentWidth = (localEnd - localStart) / localSegments;
    args.back() = localStart + segmentWidth / 2.0;

    for (int segment = 0; segment < localSegments; segment++) {
      if (limits.empty()) {
        localIntegral += integrandFunction(args) * segmentWidth;
      } else {
        localIntegral += calculateIntegral(integrandFunction, epsilon, limits, args) * segmentWidth;
      }
      args.back() += segmentWidth;
    }

    reduce(mpi_comm, localIntegral, globalIntegral, std::plus<>(), 0);

    if (mpi_comm.rank() == 0) {
      continueRefining = (std::abs(globalIntegral - previousGlobalIntegral) * (1.0 / 3.0) > epsilon);
    }
    broadcast(mpi_comm, continueRefining, 0);

    totalProcesses *= 2;
  }

  result = (mpi_comm.rank() == 0) ? globalIntegral : 0.0;
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
  double previousValue;
  int subdivisions = 2;
  bool flag = true;

  auto [lowerBound, upperBound] = limits_.front();
  limits_.erase(limits_.begin());
  args_.emplace_back(0.0);

  while (flag) {
    previousValue = integralValue;
    integralValue = 0.0;

    double segmentWidth = (upperBound - lowerBound) / subdivisions;
    args_.back() = lowerBound + segmentWidth / 2.0;

    for (int i = 0; i < subdivisions; ++i) {
      if (limits_.empty()) {
        integralValue += func_(args_) * segmentWidth;
      } else {
        integralValue += calculateIntegral(func_, epsilon_, limits_, args_) * segmentWidth;
      }
      args_.back() += segmentWidth;
    }

    subdivisions *= 2;

    flag = (std::abs(integralValue - previousValue) > epsilon_);
  }

  args_.pop_back();
  limits_.insert(limits_.begin(), {lowerBound, upperBound});

  return integralValue;
}

}  // namespace korneeva_e_rectangular_integration_method_mpi
