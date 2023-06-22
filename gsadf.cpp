#include "gsadf.h"

int default_win_size(int obs)
{
    int r0 = 0.01 + 1.8 / sqrt(obs);
    return r0;
}

std::tuple<MatrixXd, VectorXd> get_reg(VectorXd &y, int adflag, bool null_hyp = false)
{

    VectorXd dy = y.segment(1, y.size() - 1) - y.segment(0, y.size() - 1);

    
    VectorXd dy01 = dy.segment(adflag, dy.size() - adflag);

    int start_row = null_hyp ? 0 : 1;
    int ncol = y.size() - 1 - adflag;

    MatrixXd reg = MatrixXd::Constant(start_row + adflag+1, ncol, NAN);

    for (int i = 0; i < ncol; i++)
    {
        if (null_hyp == false)
            reg(0, i) = y(adflag + i);
        

        reg(start_row, i) = 1;
        for (int j = 1; j <= adflag; j++)
        {
            reg(start_row + j, i) = dy(adflag  + i - j);

        }
    }

    return (std::make_tuple(reg, dy01));
}

void rls_multi(VectorXd &y, MatrixXd &t_x, jlcxx::ArrayRef<double> badf, jlcxx::ArrayRef<double> bsadf, int min_win)
{

    int obs = y.size();  


    int x_width = t_x.rows(); // the number of independent variables

    VectorXd temp_t = VectorXd::Constant(bsadf.size(), NAN);

    

    // g is a square matrix of size x_width, b is a column vector of size x_width
    MatrixXd g = MatrixXd::Constant(x_width, x_width, NAN);
    MatrixXd b = VectorXd::Constant(x_width, NAN);

    for (int r2 = min_win - 1; r2 < obs; r2++)
    {
        for (int r1 = r2 - min_win + 1; r1 >= 0; r1--)
        {
            
            int win_size = r2 - r1 + 1;
            Ref<MatrixXd> tsx = t_x.middleCols(r1 , win_size);
            Ref<VectorXd> sy = y.segment(r1, win_size);
            
            if (win_size == min_win)
            {
                g = (tsx * tsx.transpose()).inverse();
                b = g * tsx * sy;
                
            }
            else
            {
                MatrixXd t_new_x = t_x.col(r1);
                MatrixXd gx = g * t_new_x;
                
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

void rls_gsadf(jlcxx::ArrayRef<double> y_ori, int adflag, int min_win, jlcxx::ArrayRef<double> badf, jlcxx::ArrayRef<double> bsadf)
{

    VectorXd y = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(y_ori.data(), y_ori.size());
    int obs_ori = y.size();
    
    
    if (adflag > 0)
    {
        MatrixXd t_x;
        VectorXd dy01;

        std::tie(t_x, dy01) = get_reg(y, adflag, false);
        
        rls_multi(dy01, t_x, badf, bsadf, min_win);
    }
    else
    {
        VectorXd x = y.segment(0, obs_ori - 1);                            // lag y
        VectorXd dy01 = y.segment(1, obs_ori-1) - y.segment(0, obs_ori - 1); // diff(y)
        rls_zero(dy01, x, badf, bsadf, min_win);
    }

    //return std::make_tuple(badf, bsadf);
}

void rls_zero(VectorXd &y, VectorXd &x, jlcxx::ArrayRef<double> badf, jlcxx::ArrayRef<double> bsadf, int min_win)
{
    int obs = y.size();

    VectorXd x_x = x.array() * x.array();
    VectorXd x_y = x.array() * y.array();

    VectorXd temp_t = VectorXd::Constant(bsadf.size(), NAN);

    double sumx, sumy, sumxx, sumxy;
    double meanx, meany;
    double den, alpha, beta, sbeta;

    for (int r2 = min_win - 1; r2 < obs; r2++)
    {
        sumx = x.segment(r2 - min_win + 1, min_win).sum();
        sumy = y.segment(r2 - min_win + 1, min_win).sum();
        sumxx = x_x.segment(r2 - min_win + 1, min_win).sum();
        sumxy = x_y.segment(r2 - min_win + 1, min_win).sum();
        
        for (int r1 = r2 - min_win + 1; r1 >= 0; r1--)
        {
            int win_size = r2 - r1 + 1;
            if (win_size != min_win)
            {
                sumx += x(r1);
                sumy += y(r1);
                sumxx += x_x(r1);
                sumxy += x_y(r1);
            }
            meanx = sumx / win_size;
            meany = sumy / win_size;
            den = sumxx / win_size - meanx * meanx;

            beta = (sumxy / win_size - meanx * meany) / den;

            alpha = meany - beta * meanx;

            VectorXd u = y.segment(r1, win_size).array() - alpha;
            u -= beta * x.segment(r1, win_size);
            VectorXd suu = u.array() * u.array();
            sbeta = sqrt(suu.sum() / (win_size - 2) / den / win_size);
            temp_t(r1) = beta / sbeta;
            if (r1 == 0)
                badf[r2 - min_win + 1] = temp_t(r1);
        }
        bsadf[r2 - min_win + 1] = temp_t.segment(0, r2 - min_win + 2).maxCoeff();
    }
}
