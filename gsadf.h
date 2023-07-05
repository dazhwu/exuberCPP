#pragma once

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <tuple>
#include <vector>
#include <Eigen/Dense>
// #include "jlcxx/jlcxx.hpp"

#ifdef __cplusplus
extern "C"
{
#endif

  using std::cout;
  using std::endl;
  using std::string;
  using std::tuple;
  using std::vector;

  using Eigen::ArrayXd;
  using Eigen::Map;
  using Eigen::MatrixXd;
  using Eigen::Ref;
  using Eigen::VectorXd;

  int default_win_size(int obs);
  void rls_multi(const Ref<const VectorXd> &, const Ref<const MatrixXd> &, double, double *, int, int);
  // void rls_zero(const Ref<const VectorXd> &, const Ref<const VectorXd> &, double *, double *, int, int);
  void rls_zero(const Ref<const ArrayXd> &, const Ref<const ArrayXd> &, double *, double *, int, int);
  void rls_gsadf(double *, int, int, int, double *, double *);
  std::tuple<MatrixXd, VectorXd> get_reg(const Ref<const VectorXd> &, int adflag);

#ifdef __cplusplus
}

#endif
