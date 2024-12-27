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
      : Task(std::move(taskData)), integrandFunction(std::move(func)) {}

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
      : Task(std::move(taskData)), integrandFunction(std::move(func)) {}

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

double calculateIntegral(const Function& func, double epsilon, std::vector<std::pair<double, double>> limits,
                         std::vector<double> args);

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

  double globalRes = 0.0;
  double prevGlobalRes = 0.0;
  double localRes = 0.0;
  int numProcs = mpi_comm.size();

  auto [start, end] = limits.front();
  double step = (end - start) / numProcs;

  double localStart = start + step * mpi_comm.rank();
  double localEnd = localStart + step;

  limits.erase(limits.begin());
  args.emplace_back(0.0);

  bool refine = true;
  while (refine) {
    prevGlobalRes = globalRes;
    globalRes = 0.0;
    localRes = 0.0;

    int localSegs = numProcs / mpi_comm.size();
    double segWidth = (localEnd - localStart) / localSegs;
    args.back() = localStart + segWidth / 2.0;

    for (int i = 0; i < localSegs; i++) {
      if (limits.empty()) {
        localRes += integrandFunction(args) * segWidth;
      } else {
        localRes += calculateIntegral(integrandFunction, epsilon, limits, args) * segWidth;
      }
      args.back() += segWidth;
    }

    reduce(mpi_comm, localRes, globalRes, std::plus<>(), 0);

    if (mpi_comm.rank() == 0) {
      refine = (std::abs(globalRes - prevGlobalRes) * (1.0 / 3.0) > epsilon);
    }
    broadcast(mpi_comm, refine, 0);

    numProcs *= 2;
  }

  result = (mpi_comm.rank() == 0) ? globalRes : 0.0;
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
double calculateIntegral(const Function& func, double epsilon, std::vector<std::pair<double, double>> limits,
                         std::vector<double> args) {
  double integralValue = 0;
  double prevValue = 0;
  int subdivisions = 2;
  bool flag = true;

  auto [low, high] = limits.front();
  limits.erase(limits.begin());
  args.push_back(0.0);

  while (flag) {
    prevValue = integralValue;
    integralValue = 0.0;

    double step = (high - low) / subdivisions;
    args.back() = low + step / 2.0;

    for (int i = 0; i < subdivisions; ++i) {
      if (limits.empty()) {
        integralValue += func(args) * step;
      } else {
        integralValue += calculateIntegral(func, epsilon, limits, args) * step;
      }
      args.back() += step;
    }

    subdivisions *= 2;

    flag = (std::abs(integralValue - prevValue) > epsilon);
  }

  args.pop_back();
  limits.insert(limits.begin(), {low, high});

  return integralValue;
}

}  // namespace korneeva_e_rectangular_integration_method_mpi
