from operator import itemgetter
import numpy as np
import pandas as pd
import random
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler


def point_segment_distance(point_x, point_y, seg_x1, seg_y1, seg_x2, seg_y2):
    c = seg_x2 - seg_x1
    d = seg_y2 - seg_y1
    param = (((point_x - seg_x1)*c + (point_y - seg_y1)*d)/(c**2 + d**2) * (1 - (seg_x1 == seg_x2) * (seg_y1 == seg_y2))
             - (seg_x1 == seg_x2) * (seg_y1 == seg_y2))
    xx = seg_x1 * (param < 0) + seg_x2 * (param > 1) + (seg_x1 + param * c) * (param >= 0) * (param <= 1)
    yy = seg_y1 * (param < 0) + seg_y2 * (param > 1) + (seg_y1 + param * d) * (param >= 0) * (param <= 1)
    return np.sqrt((point_x - xx)**2 + (point_y - yy)**2)


def renorm_angle(a):
    # ASSUMES INPUT IN DEGREES
    return (a + 180) % 360 - 180


def vector_angle(v_x, v_y):
    # OUTPUT IN DEGREES
    return renorm_angle(180 / np.pi * np.atan2(v_y, v_x))


def point_in_fov(viewer_x, viewer_y, view_dir, obj_x, obj_y, fov):
    # ASSUMES INPUT IN DEGREES
    return 2 * np.abs(renorm_angle(view_dir - vector_angle(obj_x - viewer_x, obj_y - viewer_y))) < fov


def circular_mean(angles):
    # ASSUMES INPUT IN DEGREES
    return 180/np.pi * np.atan2(np.sin(np.pi/180 * angles).sum(), np.cos(np.pi/180 * angles).sum())


def root():
    return 'nfl-big-data-bowl-2025\\'


def gen_uuid_col(df):
    df['uuid'] = df['gameId']*10000 + df['playId']
    return df


def gen_fuid_col(df):
    df['fuid'] = df['uuid']*1000 + df['frameId']
    return df


def filter_play_uuid(df, uuids):
    return df[~df['uuid'].isin(uuids)].reset_index(drop=True)


def corrected_players(players):
    return players


def corrected_games(games):
    return games


def corrected_plays(plays):
    plays.loc[(plays['gameId'] == 2022092900) & (plays['playId'] == 3501), 'passTippedAtLine'] = False
    plays.loc[(plays['gameId'] == 2022100209) & (plays['playId'] == 290), 'passTippedAtLine'] = False
    plays.loc[(plays['gameId'] == 2022101606) & (plays['playId'] == 3449), 'playDescription'] = (
        plays.loc[(plays['gameId'] == 2022101606) & (plays['playId'] == 3449), 'playDescription'].values[0].replace('pass intended', 'pass short right intended')
    )
    return plays


def corrected_player_play(pp):
    return pp


def corrected_tracking(tracking):
    # MISMATCHED FRAME IDS
    tracking.loc[
        (tracking['gameId'] == 2022092508) & (tracking['playId'] == 3104) & (tracking['frameId'] > 189),
        'frameId'
    ] -= 6

    # NO PRE SNAP FRAMES
    uuid_list = [20220915000281, 20220925064285, 20221016113842, 20221023103990, 20221106061677]
    dummy_rows = tracking.loc[
        (tracking['frameType'] == 'SNAP') & (tracking['gameId']*10000 + tracking['playId']).isin(uuid_list)
    ]
    dummy_rows.loc[:, 'frameType'] = 'BEFORE_SNAP'
    dummy_rows.loc[:, ['frameId', 's', 'a', 'dis']] = 0
    df_list = [tracking.iloc[:dummy_rows.index[0]]]
    for i in range(len(dummy_rows)-1):
        df_list.extend([dummy_rows.iloc[i:i+1], tracking.iloc[dummy_rows.index[i]:dummy_rows.index[i+1]]])
    df_list.extend([dummy_rows.iloc[-1:], tracking.iloc[dummy_rows.index[-1]:]])
    tracking = pd.concat(df_list, ignore_index=True)
    tracking.loc[(tracking['gameId']*10000 + tracking['playId']).isin(uuid_list), 'frameId'] += 1

    # DISPLACEMENT RECALC
    tracking['dis'] = np.where(
        tracking['frameId'] > 1,
        ((tracking['x']-tracking['x'].shift(1)) ** 2 + (tracking['y']-tracking['y'].shift(1)) ** 2) ** 0.5,
        tracking['dis']
    )

    # BAD EVENT LABELS
    shovel_fuids = [20221009052736101, 20220918050100121, 20220918043526104, 20221106021014063, 20221009052736101]
    forward_fuids = [20221002043094132, 20221030023817118, 20220911011122146, 20220918032546152, 20220925013848059, 20220925001906143, 20221009131751184, 20221016032511163, 20221024002793143, 20221023051951065, 20221023041990199, 20221031002813109, 20221030111321136, 20221030070700184, 20221030071220136, 20221030071296138, 20221030002295118, 20221106103991150, 20221106001934156, 20221030023817119]
    tipped_fuids = [20221030023817119, 20220911011122152, 20220918032546153, 20220925013848060, 20220925001906146, 20221009131751185, 20221016032511166, 20221023051951069, 20221023041990200, 20221031002813114, 20221030071220141, 20221030071296141, 20221106103991151, 20221106001934160, 20221030023817122]
    empty_fuids = [20221030023817115, 20220911011122144, 20221009052736087, 20220925044168137, 20221030023817095, 20221030023817096, 20221030023817115, 20221030023817136]
    incomplete_fuids = [20221030023817136]
    tracking.loc[(tracking['gameId']*10000000+tracking['playId']*1000+tracking['frameId']).isin(shovel_fuids), 'event'] = 'pass_shovel'
    tracking.loc[(tracking['gameId']*10000000+tracking['playId']*1000+tracking['frameId']).isin(forward_fuids), 'event'] = 'pass_forward'
    tracking.loc[(tracking['gameId']*10000000+tracking['playId']*1000+tracking['frameId']).isin(tipped_fuids), 'event'] = 'pass_tipped'
    tracking.loc[(tracking['gameId']*10000000+tracking['playId']*1000+tracking['frameId']).isin(incomplete_fuids), 'event'] = 'pass_outcome_incomplete'
    tracking.loc[(tracking['gameId']*10000000+tracking['playId']*1000+tracking['frameId']).isin(empty_fuids), 'event'] = np.nan

    # PRE PLAY FRAGMENT AND/OR HUDDLE INCLUDED
    team_x = tracking.loc[
        (tracking['club'] != 'football') & (tracking['frameType'] == 'BEFORE_SNAP')
    ].groupby(['club', 'gameId', 'playId'])['x'].mean()
    team_x = team_x.groupby(['gameId', 'playId']).agg(['idxmin', 'idxmax'])
    offense = tracking['club'] == np.where(
        tracking['playDirection'] == 'right',
        tracking.join(team_x, on=['gameId', 'playId'])['idxmin'].apply(itemgetter(0)),
        tracking.join(team_x, on=['gameId', 'playId'])['idxmax'].apply(itemgetter(0))
    )
    snap_xy = tracking.join(tracking.loc[
                      tracking['frameType'] == 'SNAP', ['gameId', 'playId', 'nflId', 'x', 'y']
                  ].set_index(['gameId', 'playId', 'nflId']), on=['gameId', 'playId', 'nflId'], rsuffix='_snap'
                            )[['x_snap', 'y_snap']].rename(columns={'x_snap': 'x', 'y_snap': 'y'})
    far_to_snap = (((snap_xy - tracking[['x', 'y']])**2).sum(axis=1)**0.5 > 2).rename('far')
    data_group = tracking.loc[
        offense & (tracking['frameType'] == 'BEFORE_SNAP')
    ].join(far_to_snap).groupby(['gameId', 'playId', 'frameId'])
    hud = (data_group[['far']].sum() >= 7).join((data_group[['x', 'y']].std() < 2).all(axis=1).rename('h')).all(axis=1)
    fragments = hud.loc[hud].reset_index().groupby(['gameId', 'playId'])['frameId'].max().reset_index()
    fragments = gen_uuid_col(fragments)
    remove_frames = tracking['frameId'] <= tracking.join(
        fragments.set_index(['gameId', 'playId']), on=['gameId', 'playId'], rsuffix='_hud'
    )['frameId_hud']
    tracking = tracking.drop(remove_frames.loc[remove_frames].index)
    tracking.loc[(tracking['gameId']*10000+tracking['playId']).isin(fragments['uuid']), 'frameId'] = (
        tracking['frameId'] - tracking.join(
            fragments.set_index(['gameId', 'playId']), on=['gameId', 'playId'], rsuffix='_hud'
        )['frameId_hud']
    )
    tracking.loc[
        (tracking['gameId'] * 10000 + tracking['playId']).isin(fragments['uuid']) & (tracking['frameId'] == 1), 'event'
    ] = 'huddle_break_offense'

    return tracking


def players_data_pipeline():
    players = pd.read_csv(root() + 'players.csv')
    players = corrected_players(players)
    return players


def games_data_pipeline():
    games = pd.read_csv(root() + 'games.csv')
    games = corrected_games(games)
    return games


def plays_data_pipeline():
    plays = pd.read_csv(root() + 'plays.csv')
    plays = corrected_plays(plays)
    plays = gen_uuid_col(plays)
    return plays


def pp_data_pipeline():
    pp = pd.read_csv(root() + 'player_play.csv')
    pp = corrected_player_play(pp)
    pp = gen_uuid_col(pp)
    return pp


def tracking_pipeline():
    tracking = pd.concat(
        [pd.read_csv(root() + f'tracking_week_{n}.csv') for n in range(1, 10)], axis=0
    ).reset_index(drop=True)
    tracking = corrected_tracking(tracking)
    tracking = gen_uuid_col(tracking)
    tracking = gen_fuid_col(tracking)
    return tracking


def master_data_pipeline():
    games = games_data_pipeline()
    players = players_data_pipeline()
    plays = plays_data_pipeline()
    pp = pp_data_pipeline()
    tracking = tracking_pipeline()

    pass_uuid = plays.loc[
        plays['passResult'].isin(['C', 'I', 'IN'])
        & (~(plays['qbSpike'].isna() | plays['qbSpike']))
        & (~plays['playDescription'].str.contains('Illegal Forward Pass'))
        & (~plays['uuid'].isin([20220911034434]))
        & (~plays['uuid'].isin(tracking.loc[tracking['event'] == 'lateral', 'uuid'])), 'uuid']
    pass_plays = plays.loc[plays['uuid'].isin(pass_uuid)]

    pass_plays['passerId'] = pd.merge(
        pass_plays['playDescription'].apply(lambda x: x[x.index(' pass ')-1::-1][x[x.index(' pass ')-1::-1].index(' ')-1::-1]),
        pd.read_csv('qb_ids.csv'),
        left_on='playDescription', right_on='plyr', how='left'
    )['passerId'].to_numpy()
    pass_uuid = np.setdiff1d(pass_uuid, pass_plays.loc[pass_plays['passerId'].isna(), 'uuid'])
    pass_plays[['passDepth', 'passSide']] = np.array(pass_plays['playDescription'].apply(
        lambda x: [i for i in x[x.index(' pass ')+6:].replace('.', '').split(' ')[:3] if i not in ['to', 'intended', 'incomplete', 'INTERCEPTED']]
    ).to_list())
    pass_plays = pass_plays.join(tracking.groupby('uuid')['playDirection'].first(), on='uuid')

    pass_plays = pass_plays.loc[pass_plays['uuid'].isin(pass_uuid)].reset_index(drop=True)
    pass_pp = pp.loc[pp['uuid'].isin(pass_uuid)].reset_index(drop=True)
    pass_tracking = tracking.loc[tracking['uuid'].isin(pass_uuid)].reset_index(drop=True)
    pass_plays['passerId'] = pass_plays['passerId'].astype(int)

    pass_plays.loc[pass_plays['passerId'] == 52413, 'dropbackType'] = pass_plays['dropbackType'].str.replace('RIGHT', '^^^^')
    pass_plays.loc[pass_plays['passerId'] == 52413, 'dropbackType'] = pass_plays['dropbackType'].str.replace('LEFT', 'RIGHT')
    pass_plays.loc[pass_plays['passerId'] == 52413, 'dropbackType'] = pass_plays['dropbackType'].str.replace('^^^^', 'LEFT')
    pass_plays.loc[pass_plays['passerId'] == 52413, 'targetY'] = 53.33 - pass_plays['targetY']
    pass_tracking.loc[pass_tracking['uuid'].isin(pass_plays.loc[pass_plays['passerId'] == 52413, 'uuid']), 'y'] = 53.33 - pass_tracking['y']
    pass_tracking.loc[pass_tracking['uuid'].isin(pass_plays.loc[pass_plays['passerId'] == 52413, 'uuid']), ['o', 'dir']] = (pass_tracking[['o', 'dir']] + 180) % 360
    pass_plays.loc[pass_plays['playDirection'] == 'left', ['absoluteYardlineNumber', 'targetX']] = 120 - pass_plays[['absoluteYardlineNumber', 'targetX']]
    pass_plays.loc[pass_plays['playDirection'] == 'left', 'targetY'] = 53.33 - pass_plays['targetY']
    pass_tracking.loc[pass_tracking['playDirection'] == 'left', 'x'] = 120 - pass_tracking['x']
    pass_tracking.loc[pass_tracking['playDirection'] == 'left', 'y'] = 53.33 - pass_tracking['y']
    pass_tracking.loc[pass_tracking['playDirection'] == 'left', ['o', 'dir']] = (pass_tracking[['o', 'dir']] + 180) % 360
    pass_tracking[['o', 'dir']] = (90 - pass_tracking[['o', 'dir']]) % 360

    test = pd.merge(pass_tracking.loc[pass_tracking['club'] == 'football'].join(pass_plays[['uuid', 'passerId']].set_index('uuid'), on='uuid'), pass_tracking, left_on=['fuid', 'passerId'], right_on=['fuid', 'nflId'], suffixes=('', '_qb'))
    test['dist'] = np.sqrt((test['x'] - test['x_qb'])**2 + (test['y'] - test['y_qb'])**2)
    test['acc'] = np.where(test['uuid'] == test['uuid'].shift(-1), test['s'].shift(-1) - test['s'], np.nan)
    test = test.join(test.loc[test['event'].isin(['pass_forward', 'pass_shovel']), ['uuid', 'frameId']].rename(columns={'frameId': 'passFrame'}).set_index('uuid'), on='uuid')
    test = test.join(test.loc[test['event'].isin(['pass_arrived', 'pass_outcome_caught', 'pass_outcome_incomplete', 'pass_outcome_interception', 'pass_outcome_touchdown'])].groupby('uuid')['frameId'].first().rename('arrivalFrame'), on='uuid')

    pass_start_fuids = test.loc[(test['acc']>=3) & ((test['frameId'] - test['passFrame']).abs() <= 5)].groupby('uuid').first()['fuid'].values
    pass_start_fuids = np.array([*pass_start_fuids, *test.loc[~test['uuid'].isin(np.floor(pass_start_fuids/1000).astype(int)) & (test['dist'] <= 2) & ((test['frameId'] - test['passFrame']).abs() <= 3)].groupby('uuid').last()['fuid'].values])
    pass_start_fuids = np.array([*pass_start_fuids, *test.loc[~test['uuid'].isin(np.floor(pass_start_fuids/1000).astype(int)) & test['event'].isin(['pass_forward', 'pass_shovel']), 'fuid'].values])
    test = test.join(test.loc[test['fuid'].isin(pass_start_fuids), ['uuid', 'frameId']].rename(columns={'frameId': 'startFrame'}).set_index('uuid'), on='uuid')

    pass_end_fuids = test.loc[(test['acc']<=-3) & ((test['frameId'] - test['arrivalFrame']).abs() <= 5)].groupby('uuid').first()['fuid'].values
    pass_end_fuids = np.array([*pass_end_fuids, *test.loc[~test['uuid'].isin(np.floor(pass_end_fuids/1000).astype(int)) & (test['frameId'] == test['arrivalFrame']), 'fuid'].values])
    pass_end_fuids = np.array([*pass_end_fuids, *test.loc[~test['uuid'].isin(np.floor(pass_end_fuids/1000).astype(int)) & (test['acc']<=-3) & (test['frameId'] > test['startFrame'])].groupby('uuid').first()['fuid'].values])
    pass_end_fuids = np.array([*pass_end_fuids, *np.minimum(test.loc[~test['uuid'].isin(np.floor(pass_end_fuids/1000).astype(int)) & (test['frameId'] == test['startFrame']), 'fuid'].values + 10, test.loc[~test['uuid'].isin(np.floor(pass_end_fuids/1000).astype(int))].groupby('uuid')['fuid'].max().values)])
    test = test.join(test.loc[test['fuid'].isin(pass_end_fuids), ['uuid', 'frameId']].rename(columns={'frameId': 'endFrame'}).set_index('uuid'), on='uuid')

    trajectory = test.loc[(test['frameId'] > test['startFrame']) & (test['frameId'] <= test['endFrame']), ['uuid', 'x', 'y']].join(test.loc[test['frameId']==test['startFrame'], ['uuid', 'x_qb', 'y_qb']].set_index('uuid'), on='uuid')
    trajectory = trajectory.join(pass_plays[['uuid', 'absoluteYardlineNumber', 'passLength']].set_index('uuid'), on='uuid')
    trajectory['targetX'] = trajectory['absoluteYardlineNumber'] + trajectory['passLength']
    trajectory['theta'] = np.arctan((trajectory['y'] - trajectory['y_qb'])/(trajectory['x'] - trajectory['x_qb']))
    trajectory = trajectory.groupby('uuid')[['x_qb', 'y_qb', 'theta', 'targetX']].median()
    trajectory['targetY'] = np.tan(trajectory['theta']) * (trajectory['targetX'] - trajectory['x_qb']) + trajectory['y_qb']

    pass_plays[['targetX', 'targetY']] = pass_plays[['targetX', 'targetY']].fillna(pass_plays.join(trajectory.loc[~trajectory['targetX'].isin([10, 110])], on='uuid', lsuffix='_')[['targetX', 'targetY']])
    pass_plays[['targetX', 'targetY']] = pass_plays[['targetX', 'targetY']].fillna(pass_plays.join(test.loc[test['frameId'] == test['endFrame'], ['uuid', 'x', 'y']].set_index('uuid'), on='uuid')[['x', 'y']].rename(columns={'x': 'targetX', 'y': 'targetY'}))
    pass_plays = pass_plays.join(pd.DataFrame(np.array([np.floor(pass_start_fuids/1000), pass_start_fuids % 1000]).T, columns=['uuid', 'startFrame'], dtype=int).set_index('uuid'), on='uuid')
    pass_plays = pass_plays.join(pd.DataFrame(np.array([np.floor(pass_end_fuids/1000), pass_end_fuids % 1000]).T, columns=['uuid', 'endFrame'], dtype=int).set_index('uuid'), on='uuid')

    nearest_wr = pass_tracking.join(pass_plays[['uuid', 'targetX', 'targetY', 'endFrame']].set_index('uuid'), on='uuid')
    nearest_wr = nearest_wr.loc[(nearest_wr['frameId'] == nearest_wr['endFrame']) & pass_tracking.join(players[['nflId', 'position']].set_index('nflId'), on='nflId')['position'].isin(['TE', 'WR', 'RB', 'FB'])]
    nearest_wr['dist'] = np.sqrt((nearest_wr['x'] - nearest_wr['targetX'])**2 + (nearest_wr['y'] - nearest_wr['targetY'])**2)
    nearest_wr = nearest_wr.loc[nearest_wr.groupby('uuid')['dist'].idxmin()]
    pass_plays = pass_plays.join(pass_pp.loc[pass_pp['wasTargettedReceiver'] == 1, ['uuid', 'nflId']].rename(columns={'nflId': 'targetId'}).set_index('uuid'), on='uuid')
    pass_plays['targetId'] = pass_plays['targetId'].fillna(pass_plays.join(nearest_wr[['uuid', 'nflId']].rename(columns={'nflId': 'targetId'}).set_index('uuid'), on='uuid', lsuffix='_')['targetId']).astype(int)

    pass_tracking = pass_tracking.join(pass_plays[['uuid', 'possessionTeam', 'absoluteYardlineNumber', 'targetX', 'targetY', 'startFrame', 'endFrame', 'passerId', 'targetId']].set_index('uuid'), on='uuid')

    nearest_def = pass_tracking.loc[(pass_tracking['club'] != pass_tracking['possessionTeam']) & (pass_tracking['club'] != 'football') & (pass_tracking['frameId'] == pass_tracking['startFrame'])].join(pass_tracking.loc[(pass_tracking['targetId'] == pass_tracking['nflId']) & (pass_tracking['frameId'] == pass_tracking['startFrame']), ['uuid', 'x', 'y']].set_index('uuid'), on='uuid', rsuffix='_wr').join(pass_tracking.loc[(pass_tracking['passerId'] == pass_tracking['nflId']) & (pass_tracking['frameId'] == pass_tracking['startFrame']), ['uuid', 'x', 'y']].set_index('uuid'), on='uuid', rsuffix='_qb')
    nearest_def['dist_wr'] = np.sqrt((nearest_def['x'] - nearest_def['x_wr'])**2 + (nearest_def['y'] - nearest_def['y_wr'])**2)
    nearest_def['dist_qb'] = np.sqrt((nearest_def['x'] - nearest_def['x_qb'])**2 + (nearest_def['y'] - nearest_def['y_qb'])**2)
    nearest_def['dist_tar'] = np.sqrt((nearest_def['x'] - nearest_def['targetX'])**2 + (nearest_def['y'] - nearest_def['targetY'])**2)
    nearest_def['dist_traj'] = point_segment_distance(nearest_def['x'], nearest_def['y'], nearest_def['x_qb'], nearest_def['y_qb'], nearest_def['targetX'], nearest_def['targetY'])
    nearest_def['facing_qb'] = point_in_fov(nearest_def['x'], nearest_def['y'], nearest_def['o'], nearest_def['x_qb'], nearest_def['y_qb'], 135)
    nearest_def['downfield'] = nearest_def['x'] > 3 + nearest_def['absoluteYardlineNumber']
    pass_plays = pass_plays.join(nearest_def.loc[nearest_def.groupby('uuid')['dist_wr'].idxmin(), ['uuid', 'nflId']].rename(columns={'nflId': 'cornerId'}).set_index('uuid'), on='uuid')
    pass_plays = pass_plays.join(nearest_def.loc[nearest_def.loc[nearest_def['facing_qb']].groupby('uuid')['dist_tar'].idxmin(), ['uuid', 'nflId']].rename(columns={'nflId': 'safetyId'}).set_index('uuid'), on='uuid')
    pass_plays = pass_plays.join(nearest_def.loc[nearest_def.loc[nearest_def['facing_qb']].groupby('uuid')['dist_qb'].idxmin(), ['uuid', 'nflId']].rename(columns={'nflId': 'blitzerId'}).set_index('uuid'), on='uuid')
    pass_plays = pass_plays.join(nearest_def.loc[nearest_def.loc[~nearest_def['downfield']].groupby('uuid')['dist_traj'].idxmin(), ['uuid', 'nflId']].rename(columns={'nflId': 'dlineId'}).set_index('uuid'), on='uuid')
    pass_plays['safetyId'] = pass_plays['safetyId'].fillna(pass_plays.join(nearest_def.loc[nearest_def.groupby('uuid')['dist_tar'].idxmin(), ['uuid', 'nflId']].rename(columns={'nflId': 'safetyId'}).set_index('uuid'), on='uuid', lsuffix='_')['safetyId'])
    pass_plays['blitzerId'] = pass_plays['blitzerId'].fillna(pass_plays.join(nearest_def.loc[nearest_def.groupby('uuid')['dist_qb'].idxmin(), ['uuid', 'nflId']].rename(columns={'nflId': 'blitzerId'}).set_index('uuid'), on='uuid', lsuffix='_')['blitzerId'])
    pass_plays['dlineId'] = pass_plays['dlineId'].fillna(pass_plays.join(nearest_def.loc[nearest_def.groupby('uuid')['dist_traj'].idxmin(), ['uuid', 'nflId']].rename(columns={'nflId': 'dlineId'}).set_index('uuid'), on='uuid', lsuffix='_')['dlineId'])
    pass_plays[['cornerId', 'safetyId', 'blitzerId', 'dlineId']] = pass_plays[['cornerId', 'safetyId', 'blitzerId', 'dlineId']].astype(int)
    pass_tracking = pass_tracking.join(pass_plays[['uuid', 'cornerId', 'safetyId', 'blitzerId', 'dlineId']].set_index('uuid'), on='uuid')

    yac_cone = pass_tracking.loc[(pass_tracking['frameId'] == pass_tracking['startFrame']) & (pass_tracking['club'] != 'football') & (pass_tracking['nflId'] != pass_tracking['targetId'])]
    yac_cone = yac_cone.join(pd.DataFrame(np.median(np.concatenate((vector_angle((nearest_def['targetX'] - nearest_def['x_qb']).groupby(nearest_def['uuid']).first(), (nearest_def['targetY'] - nearest_def['y_qb']).groupby(nearest_def['uuid']).first()).to_numpy().reshape((-1, 1)), -(90-135/2)*np.ones((len(pass_plays), 1)), (90-135/2)*np.ones((len(pass_plays), 1))), axis=1), axis=1), columns=['o_yac'], index=nearest_def.groupby('uuid').first().index), on='uuid')
    yac_cone['in_cone'] = point_in_fov(yac_cone['targetX'], yac_cone['targetY'], yac_cone['o_yac'], yac_cone['x'], yac_cone['y'], 135)
    yac_cone['onPosTeam'] = yac_cone['possessionTeam'] == yac_cone['club']
    yac_cone = yac_cone.groupby(['uuid', 'onPosTeam'])['in_cone'].sum().reset_index().set_index('uuid')
    pass_plays = pass_plays.join(yac_cone.loc[yac_cone['onPosTeam'], 'in_cone'].rename('offPlayersDownfield'), on='uuid')
    pass_plays = pass_plays.join(yac_cone.loc[~yac_cone['onPosTeam'], 'in_cone'].rename('defPlayersDownfield'), on='uuid')

    pass_plays = pass_plays.join((pass_tracking.loc[(pass_tracking['club'] == 'football') & (pass_tracking['frameId'] >= pass_tracking['startFrame']) & (pass_tracking['frameId'] <= pass_tracking['endFrame']), 's'] >= pass_tracking.loc[(pass_tracking['club'] == 'football') & (pass_tracking['frameId'] >= pass_tracking['startFrame']) & (pass_tracking['frameId'] <= pass_tracking['endFrame'])].groupby('uuid')['s'].transform('max')*0.8).groupby(pass_tracking['uuid']).sum().rename('hangTime') * 0.1, on='uuid')
    pass_plays['hangTime'] = np.round(pass_plays['hangTime'].fillna(0.1), 1)
    pass_plays = pass_plays.join(pass_tracking.loc[(pass_tracking['club'] == 'football') & (pass_tracking['frameId'] >= pass_tracking['startFrame']) & (pass_tracking['frameId'] <= pass_tracking['endFrame'])].groupby('uuid')['s'].max().rename('throwVelo'), on='uuid')
    pass_plays['throwVelo'] = pass_plays['throwVelo'].fillna(0)
    pass_plays = pass_plays.join(trajectory['theta'].rename('throwAngle') * 180/np.pi, on='uuid')
    pass_plays['throwAngle'] = pass_plays['throwAngle'].fillna(0)

    for key, val in {'passerId': '_qb', 'targetId': '_wr', 'cornerId': '_cb', 'safetyId': '_sf', 'dlineId': '_dl', 'blitzerId': '_bz'}.items():
        pass_plays = pass_plays.join(pass_tracking.loc[((pass_tracking['frameId'] - pass_tracking['startFrame']).abs() <= 1) & (
                    pass_tracking['nflId'] == pass_tracking[key]), ['uuid', 'x', 'y', 's']].groupby('uuid').mean().add_suffix(val), on='uuid')
        pass_plays = pass_plays.join(pass_tracking.loc[((pass_tracking['frameId'] - pass_tracking['startFrame']).abs() <= 1) & (
                    pass_tracking['nflId'] == pass_tracking[key]), ['uuid', 'o', 'dir']].groupby('uuid').agg(circular_mean).add_suffix(val), on='uuid')

    pass_plays['posTeamIsHome'] = pass_plays.join(games.set_index('gameId'), on='gameId', lsuffix='_')['homeTeamAbbr'] == pass_plays['possessionTeam']

    pass_plays = pass_plays.join(pass_pp.groupby('uuid')[['quarterbackHit', 'wasInitialPassRusher', 'causedPressure', 'wasRunningRoute', 'motionSinceLineset']].sum().rename(columns={'quarterbackHit': 'qbHit', 'wasInitialPassRusher': 'passRushers', 'causedPressure': 'qbPressures', 'wasRunningRoute': 'routeRunners', 'motionSinceLineset': 'motionMen'}).astype(int), on='uuid')
    pass_plays = pass_plays.join(pass_pp.groupby('uuid')['timeToPressureAsPassRusher'].min().fillna(10).rename('timeToPressure'), on='uuid')

    pass_pp = pass_pp.join(players[['nflId', 'position']].set_index('nflId'), on='nflId')
    pass_plays = pass_plays.join(pass_pp['position'].isin(['DE', 'NT', 'DT']).groupby(pass_pp['uuid']).sum().rename('dlineCount'), on='uuid')
    pass_plays = pass_plays.join(pass_pp['position'].isin(['SS', 'FS', 'CB', 'DB']).groupby(pass_pp['uuid']).sum().rename('dbCount'), on='uuid')
    pass_plays = pass_plays.join(pass_pp['position'].isin(['WR']).groupby(pass_pp['uuid']).sum().rename('wrCount'), on='uuid')
    pass_plays = pass_plays.join(pass_pp['position'].isin(['RB', 'FB']).groupby(pass_pp['uuid']).sum().rename('rbCount'), on='uuid')
    pass_plays = pass_plays.join(pass_pp['position'].isin(['TE']).groupby(pass_pp['uuid']).sum().rename('teCount'), on='uuid')
    pass_plays = pass_plays.join(pass_pp['position'].isin(['MLB', 'LB', 'ILB', 'OLB']).groupby(pass_pp['uuid']).sum().rename('lbCount'), on='uuid')

    in_box = pass_tracking.join(pass_tracking.loc[(pass_tracking['frameType'] == 'BEFORE_SNAP') & (pass_tracking['club'] == 'football')].groupby('uuid')[['x', 'y']].median().rename(columns={'x': 'x_ball', 'y': 'y_ball'}), on='uuid')
    in_box_snap = in_box.loc[(in_box['frameType'] == 'SNAP') & (in_box['club'] != in_box['possessionTeam']) & (in_box['club'] != 'football')]
    in_box_pass = in_box.loc[(in_box['frameId'] == in_box['startFrame']) & (in_box['club'] != in_box['possessionTeam']) & (in_box['club'] != 'football')]
    pass_plays = pass_plays.join((((in_box_snap['y'] - in_box_snap['y_ball']).abs() < 4) & (in_box_snap['x'] - in_box_snap['x_ball'] < 5)).groupby(in_box_snap['uuid']).sum().rename('inBoxAtSnap'), on='uuid')
    pass_plays = pass_plays.join((((in_box_pass['y'] - in_box_pass['y_ball']).abs() < 4) & (in_box_pass['x'] - in_box_pass['x_ball'] < 5)).groupby(in_box_pass['uuid']).sum().rename('inBoxAtThrow'), on='uuid')

    pass_plays['half'] = np.floor((pass_plays['quarter']+1) / 2).astype(int)
    pass_plays['secondsLeftInHalf'] = (pass_plays['quarter'] % 2) * (pass_plays['quarter'] < 5) * 15 * 60 + pass_plays['gameClock'].str[:2].astype(int) * 60 + pass_plays['gameClock'].str[3:].astype(int)
    pass_plays['scoreDiff'] = (pass_plays['preSnapHomeScore'] - pass_plays['preSnapVisitorScore']) * np.where(pass_plays['posTeamIsHome'], 1, -1)

    pass_plays['completion'] = pass_plays['passResult'] == 'C'
    pass_plays['interception'] = pass_plays['passResult'] == 'IN'
    pass_plays['incompletion'] = pass_plays['passResult'] == 'I'
    pass_plays['touchdown'] = pass_plays['playDescription'].str.contains('TOUCHDOWN') & ~pass_plays['playDescription'].str.contains('INTERCEPT') & ~pass_plays['playDescription'].str.contains('FUMBLE')
    pass_plays['tip'] = pass_plays['passTippedAtLine']
    pass_plays['passYards'] = pass_plays['prePenaltyYardsGained']
    pass_plays['passYAC'] = np.where(pass_plays['completion'], pass_plays['prePenaltyYardsGained'] - pass_plays['passLength'], np.nan)

    base_design_matrix = pass_plays[[
        'half',
        'secondsLeftInHalf',
        'posTeamIsHome',
        'down',
        'yardsToGo',
        'scoreDiff',
        'absoluteYardlineNumber',
        'expectedPoints',
        'passLength',
        'playAction',
        'pff_runPassOption',
        'dropbackDistance',
        'timeToThrow',
        'timeInTackleBox',
        'timeToPressure',
        'unblockedPressure',
        'qbHit',
        'qbPressures',
        'passRushers',
        'inBoxAtSnap',
        'inBoxAtThrow',
        'dlineCount',
        'lbCount',
        'dbCount',
        'rbCount',
        'wrCount',
        'teCount',
        'routeRunners',
        'motionMen',
        'targetX',
        'targetY',
        'hangTime',
        'throwVelo',
        'throwAngle',
        'offPlayersDownfield',
        'defPlayersDownfield',
        *[x+pos for pos in ['_qb', '_wr', '_cb', '_sf', '_dl', '_bz'] for x in ['x', 'y', 's', 'dir', 'o']]
    ]]

    non_cat = base_design_matrix.columns
    cats = ['defensiveTeam', 'dropbackType', 'offenseFormation', 'receiverAlignment', 'pff_passCoverage', 'pff_manZone']
    def_cats = ['BAL', 'TRADITIONAL', 'SHOTGUN', '2x2', 'Cover-2', 'Zone']
    for col in cats:
        dummy_mat = pd.get_dummies(pass_plays[col], prefix=col)
        dummy_mat.columns = dummy_mat.columns.str.replace(' ', '')
        base_design_matrix = pd.concat([base_design_matrix, dummy_mat], axis=1)
    base_design_matrix = base_design_matrix.drop([s+'_'+def_cats[i] for i, s, in enumerate(cats)], axis=1)

    interaction_table = pd.DataFrame([[base_design_matrix.columns[i], base_design_matrix.columns[j]] for i in range(len(base_design_matrix.columns)) for j in range(i+1, len(base_design_matrix.columns))], columns=['var1', 'var2'])
    dupes = interaction_table.loc[~interaction_table['var1'].isin(non_cat) & ~interaction_table['var2'].isin(non_cat), 'var1'].str[:7] == interaction_table.loc[~interaction_table['var1'].isin(non_cat) & ~interaction_table['var2'].isin(non_cat), 'var2'].str[:7]
    interaction_table = interaction_table.drop(dupes.loc[dupes].index).reset_index(drop=True)
    interaction_table['int_name'] = 'inter_' + interaction_table.index.astype(str)

    interaction_design_matrix = base_design_matrix.copy()
    for i, row in interaction_table.iterrows():
        interaction_design_matrix[row['int_name']] = base_design_matrix[row['var1']] * base_design_matrix[row['var2']]

    base_design_matrix.loc[~pass_plays['tip'].astype(bool)].astype(float).to_csv('base_dm.csv', index=False)
    base_design_matrix.loc[~pass_plays['tip'].astype(bool) & pass_plays['completion']].astype(float).to_csv('base_dm_comp.csv', index=False)
    interaction_design_matrix.loc[~pass_plays['tip'].astype(bool)].astype(float).to_csv('inter_dm.csv', index=False)
    interaction_design_matrix.loc[~pass_plays['tip'].astype(bool) & pass_plays['completion']].astype(float).to_csv('inter_dm_comp.csv', index=False)
    pass_plays.loc[~pass_plays['tip'].astype(bool), ['completion', 'incompletion', 'interception']].astype(float).to_csv('pass_outcome.csv', index=False)
    pass_plays.loc[~pass_plays['tip'].astype(bool) & pass_plays['completion'], ['passYAC']].astype(float).to_csv('pass_yac_comp.csv', index=False)
    pass_plays.loc[~pass_plays['tip'].astype(bool) & pass_plays['completion'], ['touchdown']].astype(float).to_csv('pass_td_comp.csv', index=False)
    interaction_table.to_csv('interaction_lookup.csv', index=False)

    pass_plays.loc[~pass_plays['tip'].astype(bool), ['gameId', 'playId', 'passerId']].to_csv('pass_ids.csv', index=False)
    pass_plays.loc[~pass_plays['tip'].astype(bool) & pass_plays['completion'], ['passerId']].to_csv('pass_ids_comp.csv', index=False)


master_data_pipeline()


design_matrix = pd.read_csv('base_dm.csv')
outcome_matrix = pd.read_csv('pass_outcome.csv')
comp_design_matrix = pd.read_csv('base_dm_comp_edit.csv')
yac_matrix = pd.read_csv('pass_yac_comp_edit.csv')
td_matrix = pd.read_csv('pass_td_comp_edit.csv')
outcome_matrix_cat = pd.Series(np.where(outcome_matrix['completion'], 0, np.where(outcome_matrix['incompletion'], 1, 2)))

design_matrix_norm = pd.DataFrame(StandardScaler().fit_transform(design_matrix), columns=design_matrix.columns)
comp_design_matrix_norm = comp_design_matrix.drop(['receiverAlignment_3x0', 'receiverAlignment_3x3'], axis=1)
scaler = StandardScaler()
comp_design_matrix_norm = pd.DataFrame(scaler.fit_transform(comp_design_matrix_norm), columns=comp_design_matrix_norm.columns)

n_pass = len(design_matrix)
n_comp = len(comp_design_matrix)
n_feat = design_matrix_norm.shape[1]
n_comp_feat = comp_design_matrix_norm.shape[1]

k_folds = 10
index_order = random.sample(list(range(n_pass)), k=n_pass)
dividers = [int(np.round(k / k_folds * n_pass)) for k in range(k_folds+1)]
comp_index_order = random.sample(list(range(n_comp)), k=n_comp)
comp_dividers = [int(np.round(k / k_folds * n_comp)) for k in range(k_folds+1)]

# max_depth=3, eta=0.2, min_child_weight=3, subsample=0.5, alpha=0.03
outcome_model_history = []
for i, d in enumerate(dividers[:-1]):
    test_index = index_order[d:dividers[i+1]]
    train_x = design_matrix_norm.loc[np.setdiff1d(np.arange(n_pass), test_index)]
    train_y = outcome_matrix_cat.loc[np.setdiff1d(np.arange(n_pass), test_index)]

    outcome_model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', num_class=3,
                                  eta=0.2, gamma=0.01, max_leaves=4, min_child_weight=15, colsample_bytree=0.9)
    outcome_model.fit(train_x, train_y)
    outcome_model_history.append(outcome_model)

outcome_hat = pd.concat([
    pd.DataFrame(outcome_model_history[i].predict_proba(design_matrix_norm.loc[index_order[d:dividers[i + 1]]]))
    for i, d in enumerate(dividers[:-1])
], axis=0)
outcome_hat.index = index_order
outcome_hat = outcome_hat.sort_index()

yac_model_history = []
for i, d in enumerate(comp_dividers[:-1]):
    test_index = comp_index_order[d:dividers[i + 1]]
    train_x = comp_design_matrix_norm.loc[np.setdiff1d(np.arange(n_comp), test_index)]
    train_y = yac_matrix.loc[np.setdiff1d(np.arange(n_comp), test_index)]

    yac_model = XGBRegressor(objective='reg:absoluteerror', eta=0.2, max_leaves=4, min_child_weight=0.8, n_estimators=192)
    yac_model.fit(train_x, train_y)
    yac_model_history.append(yac_model)

yac_hat = pd.concat([
    pd.DataFrame(yac_model_history[i].predict(comp_design_matrix_norm.loc[comp_index_order[d:comp_dividers[i+1]]]))
    for i, d in enumerate(comp_dividers[:-1])
], axis=0)
yac_hat.index = comp_index_order
yac_hat = yac_hat.sort_index()

yac_hat.index = design_matrix.loc[(outcome_matrix['completion'] == 1) & (design_matrix['passLength'] + design_matrix['absoluteYardlineNumber'] < 110)].index
aug_matrix = design_matrix.loc[(outcome_matrix['completion'] == 0) | (design_matrix['passLength'] + design_matrix['absoluteYardlineNumber'] >= 110)].drop(['receiverAlignment_3x0', 'receiverAlignment_3x3'], axis=1)
aug_matrix_norm = scaler.transform(aug_matrix)
aug_yac_hat = pd.concat([pd.DataFrame(yac_model_history[i].predict(aug_matrix_norm), index=aug_matrix.index) for i in range(k_folds)])
aug_yac_hat = aug_yac_hat.groupby(aug_yac_hat.index).mean()
yac_hat = pd.concat([yac_hat, aug_yac_hat]).sort_index()

outcome_hat.to_csv('outcome_hat.csv', index=False)
yac_hat.to_csv('yac_hat.csv', index=False)
