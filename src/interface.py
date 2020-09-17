#The order of columns used for training and testing must me the same. This interface ensures this.

y_labels = [
    'Local:Base', 
    'Local:bfgs0.1',
    'Local:bfgs0.3',
    'Local:nedler'
]

x_labels = [
    'budget.used',
    'basic.dim',
    'disp.diff_mean_02',
    'disp.diff_mean_05',
    'disp.diff_mean_10',
    'disp.diff_mean_25',
    'disp.diff_median_02',
    'disp.diff_median_05',
    'disp.diff_median_10',
    'disp.diff_median_25',
    'disp.ratio_mean_02',
    'disp.ratio_mean_05',
    'disp.ratio_mean_10',
    'disp.ratio_mean_25',
    'disp.ratio_median_02',
    'disp.ratio_median_05',
    'disp.ratio_median_10',
    'disp.ratio_median_25',
    'ela_distr.kurtosis',
    'ela_distr.number_of_peaks',
    'ela_distr.skewness',
    'ela_meta.lin_simple.adj_r2',
    'ela_meta.lin_simple.coef.max',
    'ela_meta.lin_simple.coef.max_by_min',
    'ela_meta.lin_simple.coef.min',
    'ela_meta.lin_simple.intercept',
    'ela_meta.lin_w_interact.adj_r2',
    'ela_meta.quad_simple.adj_r2',
    'ela_meta.quad_simple.cond',
    'ela_meta.quad_w_interact.adj_r2',
    'ic.eps.max',
    'ic.eps.ratio',
    'ic.eps.s',
    'ic.h.max',
    'ic.m0',
    'limo.avg_length.reg',
    'limo.length.mean',
    'limo.ratio.mean',
    'nbc.dist_ratio.coeff_var',
    'nbc.nb_fitness.cor',
    'nbc.nn_nb.cor',
    'nbc.nn_nb.mean_ratio',
    'nbc.nn_nb.sd_ratio',
    'pca.expl_var.cor_init',
    'pca.expl_var.cor_x',
    'pca.expl_var.cov_init',
    'pca.expl_var.cov_x',
    'pca.expl_var_PC1.cor_init',
    'pca.expl_var_PC1.cor_x',
    'pca.expl_var_PC1.cov_init',
    'pca.expl_var_PC1.cov_x'
]

'''
old x_labels that might be needed on trained models
x_labels = [
    'basic.dim',
    'basic.objective_max',
    'basic.objective_min',
    'disp.diff_mean_02',
    'disp.diff_mean_05',
    'disp.diff_mean_10',
    'disp.diff_mean_25',
    'disp.diff_median_02',
    'disp.diff_median_05',
    'disp.diff_median_10',
    'disp.diff_median_25',
    'disp.ratio_mean_02',
    'disp.ratio_mean_05',
    'disp.ratio_mean_10',
    'disp.ratio_mean_25',
    'disp.ratio_median_02',
    'disp.ratio_median_05',
    'disp.ratio_median_10',
    'disp.ratio_median_25',
    'ela_distr.kurtosis',
    'ela_distr.number_of_peaks',
    'ela_distr.skewness',
    'ela_meta.lin_simple.adj_r2',
    'ela_meta.lin_simple.coef.max',
    'ela_meta.lin_simple.coef.max_by_min',
    'ela_meta.lin_simple.coef.min',
    'ela_meta.lin_simple.intercept',
    'ela_meta.lin_w_interact.adj_r2',
    'ela_meta.quad_simple.adj_r2',
    'ela_meta.quad_simple.cond',
    'ela_meta.quad_w_interact.adj_r2',
    'ic.eps.max',
    'ic.eps.ratio',
    'ic.eps.s',
    'ic.h.max',
    'ic.m0',
    'limo.avg_length.reg',
    'limo.length.mean',
    'limo.ratio.mean',
    'nbc.dist_ratio.coeff_var',
    'nbc.nb_fitness.cor',
    'nbc.nn_nb.cor',
    'nbc.nn_nb.mean_ratio',
    'nbc.nn_nb.sd_ratio',
    'pca.expl_var.cor_init',
    'pca.expl_var.cor_x',
    'pca.expl_var.cov_init',
    'pca.expl_var.cov_x',
    'pca.expl_var_PC1.cor_init',
    'pca.expl_var_PC1.cor_x',
    'pca.expl_var_PC1.cov_init',
    'pca.expl_var_PC1.cov_x'
]
'''