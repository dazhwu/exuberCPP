#include "gsadf.h"
/*
 g++  -O3 -Wall -shared -std=c++17 -fopenmp  -mavx2  -mfma  -march=native
 -I/mingw64/include/eigen3  -o gsadf.dll gsadf.cpp

 */

int default_win_size(int obs) {
  double r0 = 0.01 + 1.8 / sqrt(obs);
  return floor(r0 * obs);
}

std::tuple<MatrixXd, VectorXd> get_reg(const Ref<const VectorXd> &y, int adflag,
                                       bool null_hyp = false) {
  VectorXd dy = y.segment(1, y.size() - 1) - y.segment(0, y.size() - 1);

  VectorXd dy01 = dy.segment(adflag, dy.size() - adflag);

  int start_row = null_hyp ? 0 : 1;
  int ncol = y.size() - 1 - adflag;

  MatrixXd reg = MatrixXd::Constant(start_row + adflag + 1, ncol, NAN);

  for (int i = 0; i < ncol; i++) {
    if (null_hyp == false)
      reg(0, i) = y(adflag + i);

    reg(start_row, i) = 1;
    for (int j = 1; j <= adflag; j++) {
      reg(start_row + j, i) = dy(adflag + i - j);
    }
  }

  return (std::make_tuple(reg, dy01));
}

void rls_multi(const Ref<const VectorXd> &y, const Ref<const MatrixXd> &t_x,
               double *badf, double *bsadf, int min_win, int bsadf_size) {
  int obs = y.size();

  int x_width = t_x.rows(); // the number of independent variables

  VectorXd temp_t = VectorXd::Constant(bsadf_size, NAN);

  // g is a square matrix of size x_width, b is a column vector of size x_width
  MatrixXd g = MatrixXd::Constant(x_width, x_width, NAN);
  MatrixXd b = VectorXd::Constant(x_width, NAN);
  VectorXd gx = VectorXd::Constant(x_width, NAN);

  for (int r2 = min_win - 1; r2 < obs; r2++) {
    for (int r1 = r2 - min_win + 1; r1 >= 0; r1--) {
      int win_size = r2 - r1 + 1;
      const Ref<const MatrixXd> tsx = t_x.middleCols(r1, win_size);
      const Ref<const VectorXd> sy = y.segment(r1, win_size);

      if (win_size == min_win) {
        g = (tsx * tsx.transpose()).inverse();
        b = g * tsx * sy;
      } else {
        const Ref<const MatrixXd> t_new_x = t_x.col(r1);
        gx = g * t_new_x;

        double the_factor = 1 / (1 + (t_new_x.transpose() * gx)(0, 0));

        g -= the_factor * gx * gx.transpose();
        b -= g * t_new_x * ((b.transpose() * t_new_x)(0, 0) - y(r1));
      }

      VectorXd res = y.segment(r1, win_size) - tsx.transpose() * b;

      double sqres = res.transpose() * res;
      double vares = sqres / (win_size - x_width);

      double sb_1 = sqrt(vares * g(0, 0));

      temp_t(r1) = b(0) / sb_1;

      if (r1 == 0)
        badf[r2 - min_win + 1] = temp_t(r1);
    }

    bsadf[r2 - min_win + 1] = temp_t.segment(0, r2 - min_win + 2).maxCoeff();
  }
}

void rls_gsadf(double *y_ori, int obs_ori, int adflag, int min_win,
               double *badf, double *bsadf) {
  int max_start = obs_ori - adflag - min_win;
  VectorXd y = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(y_ori, obs_ori);
  // VectorXd badf_vec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(badf,
  // max_start); VectorXd bsadf_vec = Eigen::Map<Eigen::VectorXd,
  // Eigen::Unaligned>(bsadf, max_start); cout<<"------------------"<<endl;
  // cout<<"ori "<<badf_vec(5)<<endl;
  // cout <<"ori "<< badf[5] << endl;

  if (adflag > 0) {
    MatrixXd t_x;
    VectorXd dy01;
    std::tie(t_x, dy01) = get_reg(y, adflag, false);
    rls_multi(dy01, t_x, badf, bsadf, min_win, max_start);
  } else {
    ArrayXd x = y.segment(0, obs_ori - 1); // lag y
    ArrayXd dy01 =
        y.segment(1, obs_ori - 1) - y.segment(0, obs_ori - 1); // diff(y)
    rls_zero(dy01, x, badf, bsadf, min_win, max_start);
  }

  // cout << "ebd "<<badf_vec(5) << endl;
  // cout << "ebd "<<badf[5] << endl;

  // return std::make_tuple(badf, bsadf);
}

void rls_zero(const Ref<const ArrayXd> &y, const Ref<const ArrayXd> &x,
              double *badf, double *bsadf, int min_win, int bsadf_size) {
  int obs = y.size();

  ArrayXd x_x(obs);
  x_x = x * x.array();

  ArrayXd x_y(obs);
  x_y = x * y;

  //ArrayXd temp_t(bsadf_size);

  double sx, sy, sxx, sxy;
  for (int r2 = min_win - 1; r2 < obs; r2++) {
    double max_t = -10000;
    sx = x.segment(r2 - min_win + 1, min_win).sum();
    sy = y.segment(r2 - min_win + 1, min_win).sum();
    sxx = x_x.segment(r2 - min_win + 1, min_win).sum();
    sxy = x_y.segment(r2 - min_win + 1, min_win).sum();

    for (int r1 = r2 - min_win + 1; r1 >= 0; r1--) {
      int win_size = r2 - r1 + 1;
      if (win_size != min_win) {
        sx += x(r1);
        sy += y(r1);
        sxx += x_x(r1);
        sxy += x_y(r1);
      }
      double meanx = sx / win_size;
      double meany = sy / win_size;
      double den = sxx / win_size - meanx * meanx;

      double beta = (sxy / win_size - meanx * meany) / den;

      // double alpha = meany - beta * meanx;

      VectorXd u = y.segment(r1, win_size) - meany -
                   beta * (x.segment(r1, win_size) - meanx);
      //temp_t(r1) = beta / sqrt(u.dot(u) / (win_size - 2) / den / win_size);
      double temp_t = beta / sqrt(u.dot(u) / (win_size - 2) / den / win_size);
      if (max_t < temp_t)
        max_t = temp_t;
      if (r1 == 0)
        badf[r2 - min_win + 1] = temp_t;
        
        //badf[r2 - min_win + 1] = temp_t(r1);
    }
    bsadf[r2 - min_win + 1] = max_t; //temp_t.segment(0, r2 - min_win + 2).maxCoeff();
  }
}
