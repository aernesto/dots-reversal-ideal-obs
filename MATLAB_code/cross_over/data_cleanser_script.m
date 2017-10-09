H_2_file_names = ...
    {'TAFCDots_an_2Hz.mat',
    'TAFCDots_aw_2Hz.mat',
    'TAFCDots_cg_2Hz.mat',
    'TAFCDots_ev_2Hz.mat',
    'TAFCDots_jc_2Hz.mat',
    'TAFCDots_jm_2Hz.mat',
    'TAFCDots_kk_2Hz.mat',
    'TAFCDots_no_2Hz.mat',
    'TAFCDots_rg_2Hz.mat',
    'TAFCDots_sj_2Hz.mat',
    'TAFCDots_sv_2Hz.mat',
    'TAFCDots_tk_2Hz.mat',
    'TAFCDots_yf_2Hz.mat'
    };

H_tenth_filenames = ...
    {'TAFCDots_an_tenthHz.mat',
    'TAFCDots_cg_tenthHz.mat',
    'TAFCDots_ev_tenthHz.mat',
    'TAFCDots_jc_tenthHz.mat',
    'TAFCDots_jm_tenthHz.mat',
    'TAFCDots_kk_tenthHz.mat',
    'TAFCDots_no_tenthHz.mat',
    'TAFCDots_rg_tenthHz.mat',
    'TAFCDots_sj_tenthHz.mat',
    'TAFCDots_sv_tenthHz.mat',
    'TAFCDots_tk_tenthHz.mat',
    'TAFCDots_yf_tenthHz.mat'
    };

data_cleanser(H_2_file_names, '2')
data_cleanser(H_tenth_filenames,'tenth')
