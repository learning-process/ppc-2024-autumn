#include "../include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/utility.hpp>
#include <climits>
#include <cmath>
#include <cstdio>
#include <utility>

bool pikarychev_i_gauss3_vertical_mpi::TaskSeq::pre_processing() {
  internal_order_test();

  const auto& given_in = *reinterpret_cast<Image*>(taskData->inputs[0]);
  kernel = *reinterpret_cast<Kernel3x3*>(taskData->inputs[1]);

  imgin = Image::pad(given_in, padding);
  if (imgout.width != given_in.width || imgout.height != given_in.height) {
    imgout = Image::alloc(given_in.width, given_in.height);
  }

  return true;
}

bool pikarychev_i_gauss3_vertical_mpi::TaskSeq::validation() {
  internal_order_test();
  const auto& iin = *reinterpret_cast<Image*>(taskData->inputs[0]);
  const auto& iout = *reinterpret_cast<Image*>(taskData->outputs[0]);
  return iin.width == iout.width && iin.height == iout.height;
}

bool pikarychev_i_gauss3_vertical_mpi::TaskSeq::run() {
  internal_order_test();
  imgin.apply(kernel, imgout);
  return true;
}

bool pikarychev_i_gauss3_vertical_mpi::TaskSeq::post_processing() {
  internal_order_test();
  *reinterpret_cast<Image*>(taskData->outputs[0]) = imgout;
  return true;
}

bool pikarychev_i_gauss3_vertical_mpi::TaskPar::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    kernel = *reinterpret_cast<Kernel3x3*>(taskData->inputs[1]);

    imgin = *reinterpret_cast<Image*>(taskData->inputs[0]);
    if (imgout.width != imgin.width || imgout.height != imgin.height) {
      imgout = Image::alloc(imgin.width, imgin.height);
    }
  }

  return true;
}

bool pikarychev_i_gauss3_vertical_mpi::TaskPar::validation() {
  internal_order_test();
  if (world.rank() != 0) {
    return true;
  }
  const auto& iin = *reinterpret_cast<Image*>(taskData->inputs[0]);
  const auto& iout = *reinterpret_cast<Image*>(taskData->outputs[0]);
  return iin.width == iout.width && iin.height == iout.height;
}

int pikarychev_i_gauss3_vertical_mpi::TaskPar::distr(const boost::mpi::communicator& com, int total,
                                                     std::vector<int>& sizes) {
  const int delta = total / com.size();
  const int extra = total % com.size();

  sizes.resize(com.size(), delta);
  std::for_each_n(sizes.begin(), extra, [](auto& sz) { sz += 1; });

  return std::min(total, com.size());
}

bool pikarychev_i_gauss3_vertical_mpi::TaskPar::run() {
  internal_order_test();

  boost::mpi::broadcast(world, kernel, 0);

  auto dimensions = std::make_pair(imgin.width, imgin.height);
  boost::mpi::broadcast(world, dimensions, 0);
  const auto [width, height] = dimensions;

  std::vector<int> counts;
  const auto active = distr(world, width, counts);
  const auto dx = counts[world.rank()];
  const auto dp = (world.rank() == 0) ? pad : 0;

  auto sendcounts = counts;
  std::vector<int> senddispls(world.size());
  if (active > 1) {
    // capture horizontally neighbouring elements
    std::partial_sum(sendcounts.begin(), sendcounts.end() - 1, senddispls.begin() + 1);
    for (int i = 1; i < active; i++) {
      senddispls[i] -= pad;
    }

    sendcounts[0] += pad;
    for (int i = 1; i < active - 1; i++) {
      sendcounts[i] += 2 * pad;
    }
    sendcounts[active - 1] += pad;
  }

  auto part_in = Image::alloc(dx == 0 ? 0 : (dx + padding), dx == 0 ? 0 : (height + padding));
  auto part_out = Image::alloc(dx, dx == 0 ? 0 : height);

  // boost::mpi::scatterv DID NOT IMPLEMENT DISPLS FOR NON-MPI TYPES AND DID NOT STATE THIS FACT IN THE DOCUMENTATION!
  for (int i = 0; i < world.size(); i++) {
    sendcounts[i] *= sizeof(Image::Pixel);
    senddispls[i] *= sizeof(Image::Pixel);
  }
  for (size_t y = 0; y < height; y++) {
    boost::mpi::scatterv(world, reinterpret_cast<uint8_t*>(imgin.data.data() + (y * width)), sendcounts, senddispls,
                         reinterpret_cast<uint8_t*>(part_in.data.data() + ((y + 1) * part_in.width + dp)),
                         sendcounts[world.rank()], 0);
  }

  if (dx > 0) {
    part_in.apply(kernel, part_out);
  }

  for (size_t y = 0; y < height; y++) {
    boost::mpi::gatherv(world, part_out.data.data() + (y * part_out.width), part_out.width,
                        imgout.data.data() + (y * width), counts, 0);
  }

  return true;
}

bool pikarychev_i_gauss3_vertical_mpi::TaskPar::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<Image*>(taskData->outputs[0]) = imgout;
  }
  return true;
}
