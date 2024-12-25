#include "../include/ops_seq.hpp"

#include <climits>
#include <cmath>
#include <cstdio>

bool pikarychev_i_gauss3_vertical_seq::TaskSeq::pre_processing() {
  internal_order_test();

  const auto& given_in = *reinterpret_cast<Image*>(taskData->inputs[0]);
  kernel = *reinterpret_cast<Kernel3x3*>(taskData->inputs[1]);

  imgin = Image::pad(given_in, padding);
  if (imgout.width != given_in.width || imgout.height != given_in.height) {
    imgout = Image::alloc(given_in.width, given_in.height);
  }

  return true;
}

bool pikarychev_i_gauss3_vertical_seq::TaskSeq::validation() {
  internal_order_test();
  const auto& iin = *reinterpret_cast<Image*>(taskData->inputs[0]);
  const auto& iout = *reinterpret_cast<Image*>(taskData->outputs[0]);
  return iin.width == iout.width && iin.height == iout.height;
}

bool pikarychev_i_gauss3_vertical_seq::TaskSeq::run() {
  internal_order_test();
  imgin.apply(kernel, imgout);
  return true;
}

bool pikarychev_i_gauss3_vertical_seq::TaskSeq::post_processing() {
  internal_order_test();
  *reinterpret_cast<Image*>(taskData->outputs[0]) = imgout;
  return true;
}
