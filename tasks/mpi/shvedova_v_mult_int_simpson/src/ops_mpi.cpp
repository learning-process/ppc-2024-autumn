#include "mpi/shvedova_v_mult_int_simpson/include/ops_mpi.hpp"

double shvedova_v_mult_int_simpson_mpi::integrateSimpsonImpl(
    std::deque<std::pair<double, double>>& integrationLimits, std::vector<double>& currentArguments,
    const std::function<double(std::vector<double>& args)>& integrandFunction, double precision) {
  double currentResult = 0.0;
  double previousResult;
  int segmentCount = 2;

  auto [lowerBound, upperBound] = integrationLimits.front();
  integrationLimits.pop_front();
  currentArguments.push_back(0.0);

  do {
    previousResult = currentResult;
    currentResult = 0.0;

    double stepSize = (upperBound - lowerBound) / segmentCount;
    currentArguments.back() = lowerBound;

    if (integrationLimits.empty()) {
      currentResult += integrandFunction(currentArguments);
    } else {
      currentResult += integrateSimpsonImpl(integrationLimits, currentArguments, integrandFunction, precision);
    }

    for (int pointIndex = 1; pointIndex < segmentCount; pointIndex++) {
      currentArguments.back() = lowerBound + pointIndex * stepSize;
      double weight = (pointIndex % 2 == 0) ? 2 : 4;

      if (integrationLimits.empty()) {
        currentResult += weight * integrandFunction(currentArguments);
      } else {
        currentResult +=
            weight * integrateSimpsonImpl(integrationLimits, currentArguments, integrandFunction, precision);
      }
    }

    currentArguments.back() = upperBound;
    if (integrationLimits.empty()) {
      currentResult += integrandFunction(currentArguments);
    } else {
      currentResult += integrateSimpsonImpl(integrationLimits, currentArguments, integrandFunction, precision);
    }

    currentResult *= stepSize / 3.0;

    segmentCount *= 2;
  } while (std::abs(currentResult - previousResult) > precision);

  currentArguments.pop_back();
  integrationLimits.emplace_front(lowerBound, upperBound);

  return currentResult;
}

double shvedova_v_mult_int_simpson_mpi::integrateSimpsonParallel(
    boost::mpi::communicator& world, std::deque<std::pair<double, double>> integrationLimits, double precision,
    const std::function<double(std::vector<double>& args)>& integrandFunction) {
  std::vector<double> currentArguments;
  broadcast(world, integrationLimits, 0);
  broadcast(world, precision, 0);

  double globalResult = 0.0;
  double previousResult;
  double localResult;
  int segmentCount = 2 * world.size();

  auto [globalLowerBound, globalUpperBound] = integrationLimits.front();
  double globalStepSize = (globalUpperBound - globalLowerBound) / world.size();

  double localLowerBound = globalLowerBound + globalStepSize * world.rank();
  double localUpperBound = localLowerBound + globalStepSize;

  integrationLimits.pop_front();
  currentArguments.push_back(0.0);

  bool continueIterations = true;
  while (continueIterations) {
    previousResult = globalResult;
    globalResult = 0.0;
    localResult = 0.0;

    int localSegmentCount = segmentCount / world.size();
    double localStepSize = (localUpperBound - localLowerBound) / localSegmentCount;

    currentArguments.back() = localLowerBound;
    if (integrationLimits.empty()) {
      localResult += integrandFunction(currentArguments);
    } else {
      localResult += integrateSimpsonImpl(integrationLimits, currentArguments, integrandFunction, precision);
    }

    for (int pointIndex = 1; pointIndex < localSegmentCount; pointIndex++) {
      currentArguments.back() = localLowerBound + pointIndex * localStepSize;
      double weight = (pointIndex % 2 == 0) ? 2 : 4;

      if (integrationLimits.empty()) {
        localResult += weight * integrandFunction(currentArguments);
      } else {
        localResult += weight * integrateSimpsonImpl(integrationLimits, currentArguments, integrandFunction, precision);
      }
    }

    currentArguments.back() = localUpperBound;
    if (integrationLimits.empty()) {
      localResult += integrandFunction(currentArguments);
    } else {
      localResult += integrateSimpsonImpl(integrationLimits, currentArguments, integrandFunction, precision);
    }

    localResult *= localStepSize / 3.0;

    reduce(world, localResult, globalResult, std::plus<>(), 0);

    if (world.rank() == 0 && std::abs(globalResult - previousResult) <= precision) {
      continueIterations = false;
    }
    broadcast(world, continueIterations, 0);

    segmentCount *= 2;
  }

  return world.rank() == 0 ? globalResult : 0.0;
}

bool shvedova_v_mult_int_simpson_mpi::SimpsonMultIntParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* ptr = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);
    integrationLimits.assign(ptr, ptr + taskData->inputs_count[0]);
    tolerance_ = *reinterpret_cast<double*>(taskData->inputs[1]);
  }

  return true;
}

bool shvedova_v_mult_int_simpson_mpi::SimpsonMultIntParallel::validation() {
  internal_order_test();

  return (world.rank() != 0) || (taskData->inputs_count[0] > 0 && taskData->inputs.size() == 2 &&
                                 taskData->outputs_count[0] == 1 && !taskData->outputs.empty());
}

bool shvedova_v_mult_int_simpson_mpi::SimpsonMultIntParallel::run() {
  internal_order_test();

  result_ = integrateSimpsonParallel(world, integrationLimits, tolerance_, integrandFunction);

  return true;
}

bool shvedova_v_mult_int_simpson_mpi::SimpsonMultIntParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
  }

  return true;
}

double shvedova_v_mult_int_simpson_mpi::integrateSimpson(
    std::deque<std::pair<double, double>> integrationLimits, double precision,
    const std::function<double(std::vector<double>& args)>& integrandFunction) {
  std::vector<double> currentArguments;
  return integrateSimpsonImpl(integrationLimits, currentArguments, integrandFunction, precision);
}

bool shvedova_v_mult_int_simpson_mpi::SimpsonMultIntSequential::pre_processing() {
  internal_order_test();

  auto* ptr = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[0]);
  integrationLimits.assign(ptr, ptr + taskData->inputs_count[0]);
  tolerance_ = *reinterpret_cast<double*>(taskData->inputs[1]);

  return true;
}

bool shvedova_v_mult_int_simpson_mpi::SimpsonMultIntSequential::validation() {
  internal_order_test();

  return taskData->inputs_count[0] > 0 && taskData->inputs.size() == 2 && taskData->outputs_count[0] == 1 &&
         !taskData->outputs.empty();
}

bool shvedova_v_mult_int_simpson_mpi::SimpsonMultIntSequential::run() {
  internal_order_test();

  result_ = integrateSimpson(integrationLimits, tolerance_, integrandFunction);

  return true;
}

bool shvedova_v_mult_int_simpson_mpi::SimpsonMultIntSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;

  return true;
}