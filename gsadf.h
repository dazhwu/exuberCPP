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
#include "jlcxx/jlcxx.hpp"

using std::cout;
using std::endl;
using std::string;
using std::tuple;
using std::vector;

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::VectorXd;

int default_win_size(int obs);
std::tuple<MatrixXd, VectorXd> get_reg(VectorXd &y, int adflag);
void rls_multi(VectorXd &, MatrixXd &, jlcxx::ArrayRef<double>, jlcxx::ArrayRef<double>, int);
void rls_zero(VectorXd &y, VectorXd &, jlcxx::ArrayRef<double>, jlcxx::ArrayRef<double>, int);
void rls_gsadf(jlcxx::ArrayRef<double>, int, int, jlcxx::ArrayRef<double>, jlcxx::ArrayRef<double>);

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
  mod.method("rls_gsadf", &rls_gsadf);
}
