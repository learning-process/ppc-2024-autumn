#include "mpi/shvedova_v_mult_int_simpson/include/ops_mpi.hpp"

double shvedova_v_mult_int_simpson_mpi::integrateSimpson1D(double lowerBound, double upperBound, int segmentCount,
                                                           const Simpson1DFunction& integrand) {
  double stepSize = (upperBound - lowerBound) / segmentCount;
  double result = integrand(lowerBound) + integrand(upperBound);

  for (int i = 1; i < segmentCount; ++i) {
    double x = lowerBound + i * stepSize;
    result += ((i % 2 == 0) ? 2 : 4) * integrand(x);
  }

  return result * stepSize / 3.0;
}

double shvedova_v_mult_int_simpson_mpi::integrateSimpsonImpl(Limits& integrationLimits,
                                                             std::vector<double>& currentArguments,
                                                             const FunctionType& integrandFunction, double precision) {
  if (integrationLimits.empty()) {
    return integrandFunction(currentArguments);
  }

  auto [lowerBound, upperBound] = integrationLimits.front();
  integrationLimits.pop_front();
  currentArguments.push_back(0.0);

  double previousResult = 0.0;
  double currentResult = 0.0;
  int segmentCount = 2;

  do {
    previousResult = currentResult;

    currentResult = integrateSimpson1D(lowerBound, upperBound, segmentCount, [&](double x) {
      currentArguments.back() = x;

      if (integrationLimits.empty()) {
        return integrandFunction(currentArguments);
      }
      return integrateSimpsonImpl(integrationLimits, currentArguments, integrandFunction, precision);
    });

    segmentCount *= 2;
  } while (std::abs(currentResult - previousResult) > precision);

  currentArguments.pop_back();
  integrationLimits.emplace_front(lowerBound, upperBound);

  return currentResult;
}

double shvedova_v_mult_int_simpson_mpi::integrateSimpsonParallel(boost::mpi::communicator& world,
                                                                 Limits integrationLimits, double precision,
                                                                 const FunctionType& integrandFunction) {
  std::vector<double> currentArguments;
  broadcast(world, integrationLimits, 0);
  broadcast(world, precision, 0);

  double globalResult = 0.0;
  double previousResult = 0.0;
  double localResult = 0.0;
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

    localResult = integrateSimpson1D(localLowerBound, localUpperBound, segmentCount / world.size(), [&](double x) {
      currentArguments.back() = x;

      if (integrationLimits.empty()) {
        return integrandFunction(currentArguments);
      }
      return integrateSimpsonImpl(integrationLimits, currentArguments, integrandFunction, precision);
    });

    reduce(world, localResult, globalResult, std::plus<>(), 0);

    if (world.rank() == 0) {
      continueIterations = (std::abs(globalResult - previousResult) > precision);
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

double shvedova_v_mult_int_simpson_mpi::integrateSimpson(Limits integrationLimits, double precision,
                                                         const FunctionType& integrandFunction) {
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
