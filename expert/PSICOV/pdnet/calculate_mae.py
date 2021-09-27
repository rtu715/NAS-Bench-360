import numpy as np

def calculate_mae(PRED, YTRUE, pdb_list, length_dict):
    mae_lr_d8_list = np.zeros(len(PRED[:, 0, 0, 0]))
    mae_mlr_d8_list = np.zeros(len(PRED[:, 0, 0, 0]))
    mae_lr_d12_list = np.zeros(len(PRED[:, 0, 0, 0]))
    mae_mlr_d12_list = np.zeros(len(PRED[:, 0, 0, 0]))
    for i in range(0, len(PRED[:, 0, 0, 0])):
        L = length_dict[pdb_list[i]]
        PAVG = np.full((L, L), 100.0)
        # Average the predictions from both triangles
        for j in range(0, L):
            for k in range(j, L):
                PAVG[j, k] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
        # at distance 8 and separation 24
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        P = np.copy(PAVG)
        for p in range(len(Y)):
            for q in range(len(Y)):
                if q - p < 24:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
                    continue
                if Y[p, q] > 8:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
        mae_lr_d8 = np.nan
        if not np.isnan(np.abs(Y - P)).all():
            mae_lr_d8 = np.nanmean(np.abs(Y - P))
            #mae_lr_d8 = np.sqrt(np.nanmean(np.abs(Y - P) ** 2))
        # at distance 8 and separation 12
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        P = np.copy(PAVG)
        for p in range(len(Y)):
            for q in range(len(Y)):
                if q - p < 12:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
                    continue
                if Y[p, q] > 8:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
        mae_mlr_d8 = np.nan
        if not np.isnan(np.abs(Y - P)).all():
            mae_mlr_d8 = np.nanmean(np.abs(Y - P))
        # at distance 12 and separation 24
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        P = np.copy(PAVG)
        for p in range(len(Y)):
            for q in range(len(Y)):
                if q - p < 24:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
                    continue
                if Y[p, q] > 12:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
        mae_lr_d12 = np.nan
        if not np.isnan(np.abs(Y - P)).all():
            mae_lr_d12 = np.nanmean(np.abs(Y - P))
        # at distance 12 and separation 12
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        P = np.copy(PAVG)
        for p in range(len(Y)):
            for q in range(len(Y)):
                if q - p < 12:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
                    continue
                if Y[p, q] > 12:
                    Y[p, q] = np.nan
                    P[p, q] = np.nan
        mae_mlr_d12 = np.nan
        if not np.isnan(np.abs(Y - P)).all():
            mae_mlr_d12 = np.nanmean(np.abs(Y - P))
        # add to list
        mae_lr_d8_list[i] = mae_lr_d8
        mae_mlr_d8_list[i] = mae_mlr_d8
        mae_lr_d12_list[i] = mae_lr_d12
        mae_mlr_d12_list[i] = mae_mlr_d12

    return (np.nanmean(mae_lr_d8_list), np.nanmean(mae_mlr_d8_list),
            np.nanmean(mae_lr_d12_list), np.nanmean(mae_mlr_d12_list))
