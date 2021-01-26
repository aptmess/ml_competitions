# разбиваю на таблицы и применяю pd.melt, все остальное - некоторые преобразования к этому приёму
import pandas as pd
import numpy as np
file_name = 'case03_input_file..xlsx'
submission = pd.DataFrame()
df = pd.read_excel(file_name, sheet_name='Data', header=None, index_col=0)
splitted = np.array_split(df, np.int(df.shape[0] / 9))
for town in splitted:
    dummy = town[:-1].T
    towns = dummy.columns[0]
    dummy[towns] = towns
    res = pd.melt(dummy, id_vars=[towns, 'Диапазон'])
    res.columns = ['Region', 'Partner', 'Range', 'Value']
    map_dict = dict(zip(pd.unique(res['Range']), range(7)))
    inverse_map_dict = {v: k for k, v in map_dict.items()}
    res['Range'] = res['Range'].map(map_dict)
    res = res.sort_values(by=['Partner', 'Range'])
    res['Range'] = res['Range'].map(inverse_map_dict)
    submission = pd.concat((submission, res))
submission  = submission.dropna().reset_index(drop=True)
fileName = pd.DataFrame(index=np.arange(submission.shape[0]))
fileName['FileName'] = file_name
submission = pd.concat([fileName, submission], axis=1)
submission['Value'] = submission['Value'].astype(float)
submission.to_csv('submission.csv', index=False, encoding="cp1251")
