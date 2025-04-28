import numpy as np
import pandas as pd
from multiprocessing import Pool


def passer_rating(cmp_pct, int_pct, td_pct, ypa):
    return np.median([
        np.array([(cmp_pct - 0.3) * 5, (ypa - 3) / 4, td_pct * 20, 2.375 - int_pct * 25]),
        np.zeros(4),
        2.375 * np.ones(4)
    ], axis=0).sum() * 50 / 3


design_matrix = pd.read_csv('base_dm.csv')
outcome_matrix = pd.read_csv('pass_outcome.csv')
comp_design_matrix = pd.read_csv('base_dm_comp_edit.csv')
yac_matrix = pd.read_csv('pass_yac_comp_edit.csv')
td_matrix = pd.read_csv('pass_td_comp_edit.csv')

n_pass = len(design_matrix)
n_comp = len(comp_design_matrix)

outcome_hat = pd.read_csv('outcome_hat.csv')
yac_hat = pd.read_csv('yac_hat.csv')

ids_pass = pd.read_csv('pass_ids.csv')
ids_comp = pd.read_csv('pass_ids_comp_edit.csv')
comp_idx = design_matrix.loc[(outcome_matrix['completion'] == 1) & (design_matrix['passLength'] + design_matrix['absoluteYardlineNumber'] < 110)].index

yards_hat = pd.DataFrame(np.minimum(yac_hat.iloc[:, 0] + design_matrix['passLength'].values, 110 - design_matrix['absoluteYardlineNumber'].values))
td_hat = pd.DataFrame(np.maximum((np.repeat(110 - np.array([design_matrix['absoluteYardlineNumber'].values]).T - np.array([design_matrix['passLength'].values]).T - yac_hat.values, n_comp, axis=1) <= np.repeat(yac_matrix.values - yac_hat.loc[comp_idx].values, n_pass, axis=1).T).sum(axis=1)/n_comp, (design_matrix['passLength'] + design_matrix['absoluteYardlineNumber'] >= 110).values).reshape(-1, 1))

pr_play = pd.concat([outcome_hat.iloc[:, [0, -1]], td_hat.iloc[:, 0] * outcome_hat.iloc[:, 0], yards_hat.iloc[:, 0] * outcome_hat.iloc[:, 0]], axis=1).apply(lambda x: passer_rating(*x), axis=1)

cmp_pct_all = outcome_matrix['completion'].mean()
int_pct_all = outcome_matrix['interception'].mean()
td_pct_all = (td_matrix.values.sum() + ((design_matrix['passLength'] + design_matrix['absoluteYardlineNumber'] >= 110) * outcome_matrix['completion']).sum()) / n_pass
ypa_all = ((comp_design_matrix['passLength'] + yac_matrix.values.T[0]).sum() + ((design_matrix['passLength'] + design_matrix['absoluteYardlineNumber'] >= 110) * outcome_matrix['completion'] * design_matrix['passLength']).sum()) / n_pass
pr_all = passer_rating(cmp_pct_all, int_pct_all, td_pct_all, ypa_all)

cmp_pct_qb = outcome_matrix['completion'].groupby(ids_pass['passerId']).mean()
int_pct_qb = outcome_matrix['interception'].groupby(ids_pass['passerId']).mean()
td_pct_qb = ((design_matrix['passLength'] + design_matrix['absoluteYardlineNumber'] >= 110) * outcome_matrix['completion']).groupby(ids_pass['passerId']).sum().add(td_matrix['touchdown'].groupby(ids_comp['passerId']).sum(), fill_value=0) / ids_pass['passerId'].value_counts()
ypa_qb = ((design_matrix['passLength'] + design_matrix['absoluteYardlineNumber'] >= 110) * outcome_matrix['completion'] * design_matrix['passLength']).groupby(ids_pass['passerId']).sum().add((comp_design_matrix['passLength'] + yac_matrix.values.T[0]).groupby(ids_comp['passerId']).sum(), fill_value=0) / ids_pass['passerId'].value_counts()
pr_qb = pd.DataFrame(np.array([cmp_pct_qb, int_pct_qb, td_pct_qb, ypa_qb]).T).apply(lambda x: passer_rating(*x), axis=1)

cmp_pct_ea = outcome_hat.iloc[:, 0].mean()
int_pct_ea = outcome_hat.iloc[:, -1].mean()
td_pct_ea = (td_hat.iloc[:, 0] * outcome_hat.iloc[:, 0]).mean()
ypa_ea = (yards_hat.iloc[:, 0] * outcome_hat.iloc[:, 0]).mean()
pr_ea = passer_rating(cmp_pct_ea, int_pct_ea, td_pct_ea, ypa_ea)

cmp_pct_exp = outcome_hat.iloc[:, 0].groupby(ids_pass['passerId']).mean()
int_pct_exp = outcome_hat.iloc[:, -1].groupby(ids_pass['passerId']).mean()
td_pct_exp = (td_hat.iloc[:, 0] * outcome_hat.iloc[:, 0]).groupby(ids_pass['passerId']).mean()
ypa_exp = (yards_hat.iloc[:, 0] * outcome_hat.iloc[:, 0]).groupby(ids_pass['passerId']).mean()
pr_exp = pd.DataFrame(np.array([cmp_pct_exp, int_pct_exp, td_pct_exp, ypa_exp]).T).apply(lambda x: passer_rating(*x), axis=1)

# BOOTSTRAPPING
straps = 10000


def task(row):
    pass_indexes = ids_pass.loc[ids_pass['passerId'] == row['passerId']].index
    pass_sample = np.random.choice(pass_indexes, (straps, row['count']))
    pd.DataFrame([[
        row['passerId'], j,
        outcome_matrix.loc[pass_sample[j]].iloc[:, 0].mean(),
        outcome_hat.loc[pass_sample[j]].iloc[:, 0].mean(),
        outcome_matrix.loc[pass_sample[j]].iloc[:, -1].mean(),
        outcome_hat.loc[pass_sample[j]].iloc[:, -1].mean(),
        (td_matrix.loc[np.isin(comp_idx, pass_sample[j])].iloc[:, 0].sum() + ((design_matrix.loc[pass_sample[j], 'passLength'] + design_matrix.loc[pass_sample[j], 'absoluteYardlineNumber'] >= 110) * outcome_matrix.loc[pass_sample[j], 'completion']).sum()) / row['count'],
        (td_hat.iloc[pass_sample[j], 0] * outcome_hat.iloc[pass_sample[j], 0]).mean(),
        ((comp_design_matrix.loc[np.isin(comp_idx, pass_sample[j]), 'passLength'] + yac_matrix.loc[np.isin(comp_idx, pass_sample[j])].values.T[0]).sum() + ((design_matrix.loc[pass_sample[j], 'passLength'] + design_matrix.loc[pass_sample[j], 'absoluteYardlineNumber'] >= 110) * outcome_matrix.loc[pass_sample[j], 'completion'] * design_matrix.loc[pass_sample[j], 'passLength']).sum()) / row['count'],
        (yards_hat.iloc[pass_sample[j], 0] * outcome_hat.iloc[pass_sample[j], 0]).mean(),
    ] for j in range(straps)],
        columns=['nflId', 'strap', 'cmp_obs', 'cmp_exp', 'int_obs', 'int_exp', 'td_obs', 'td_exp', 'ypa_obs', 'ypa_exp']
    ).to_csv(f'bootstrap\\{row['passerId']}.csv', index=False)


if __name__ == '__main__':
    p = Pool(14)
    p.map(task, [inputs for i, inputs in ids_pass['passerId'].value_counts().reset_index().iterrows()])
    p.close()
    p.join()

    resampled_df = pd.concat([pd.read_csv(f'bootstrap\\{f}.csv') for f in ids_pass['passerId'].unique()], ignore_index=True)
    resampled_df['pr_obs'] = resampled_df.iloc[:, [2, 4, 6, 8]].apply(lambda x: passer_rating(*x), axis=1)
    resampled_df['pr_exp'] = resampled_df.iloc[:, [3, 5, 7, 9]].apply(lambda x: passer_rating(*x), axis=1)

    play_df = pd.concat([outcome_hat.iloc[:, [0, -1]], td_hat.iloc[:, 0] * outcome_hat.iloc[:, 0], yards_hat.iloc[:, 0] * outcome_hat.iloc[:, 0], pr_play], axis=1)
    play_df.columns = ['cmp_exp', 'int_exp', 'td_exp', 'ypa_exp', 'pass_rate_exp']
    summary_df = pd.DataFrame(np.array(
        [cmp_pct_qb, cmp_pct_exp, int_pct_qb, int_pct_exp, td_pct_qb, td_pct_exp, ypa_qb, ypa_exp, pr_qb, pr_exp]).T,
        columns=['cmp_obs', 'cmp_exp', 'int_obs', 'int_exp', 'td_obs', 'td_exp', 'ypa_obs', 'ypa_exp', 'pass_rate_obs', 'pass_rate_exp'])
    summary_df['att'] = ids_pass['passerId'].value_counts().sort_index().to_numpy()
    summary_df['nflId'] = cmp_pct_qb.index
    summary_df.loc[-1] = [cmp_pct_all, cmp_pct_ea, int_pct_all, int_pct_ea, td_pct_all, td_pct_ea, ypa_all, ypa_ea, pr_all, pr_ea, n_pass, -1]
    play_df.to_csv('play_pr_exp.csv', index=False)
    summary_df.to_csv('summary.csv', index=False)
    resampled_df.to_csv('resampled.csv', index=False)
