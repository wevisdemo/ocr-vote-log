
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from re import sub
from sys import argv

TITLE = ['ร้อยตํารวจตรี', 'พันเอก', 'ร้อยตํารวจเอก',
         'พันตํารวจโท', 'พลตํารวจเอก', 'พันตํารวจเอก']
REPLACE = [('นนางสาว', 'นางสาว'), ('นษาง', 'นาง'), ('นบนาย', 'นาย'),
           ('บาย', 'นาย'), ('พสตํารวจ', 'พลตำรวจ'), ('พลตำรวจตรวี', 'พลตำรวจตรี')]


def main(peplevotecsv, ocrcsv, votelogcol):
    df = pd.read_csv(peplevotecsv, skiprows=1)

    validate_df = df.iloc[:, 9:-1]\
        .replace(' ', np.nan)\
        .fillna(5).replace('-', 0)\
        .astype(np.int16)

    df = df.iloc[:, :9]

    df_ocr = pd.read_csv(ocrcsv, skiprows=1)
    df_ocr.columns = ['no', 'id', 'fullname', 'party', 'vote']
    df_ocr = df_ocr.astype(np.str_)

    df_ocr.party = df_ocr.party.apply(lambda x: fix_party(x.replace(
        'พรรศ', 'พรรค').replace('พรวศ', 'พรรค').replace('พรรค', ''), df.party.unique()))
    df_ocr.fullname = df_ocr.fullname.apply(
        lambda x: split_name(x, df.title.unique().tolist()+TITLE))
    df_ocr.vote = df_ocr.vote.apply(vote_encoder)

    party_b = df_ocr.party.isin(df.party.dropna())
    name_b = df_ocr.fullname\
        .apply(lambda x: x.split(' ')[0])\
        .apply(lambda x: (x == df.name.dropna()).sum())
    last_name_b = df_ocr.fullname.apply(last_name).isin(df.lastname.dropna())

    included_b = (name_b) | (last_name_b) | (party_b)
    df['ocr'] = match_name(df, df_ocr, included_b)
    merged_df = df.merge(df_ocr[['fullname', 'vote']],
                         left_on='ocr', right_index=True)

    votelog_val = validate_df.loc[merged_df.index][votelogcol]
    not_eq = merged_df[(votelog_val == merged_df.vote) == False]
    if (not not_eq.empty):
        print(not_eq)


def similar(a, b):
    return SequenceMatcher(lambda x: x in ' ', a, b).ratio()


def fix_party(p, ps):
    if not p:
        return p

    for x in ps:
        sim = similar(p, str(x))
        if sim > .8:
            return x
    return p


def split_name(name, titles):
    name = sub('[!-~]', '', name)
    for o, n in REPLACE:
        if name.startswith(o):
            name = name.replace(o, n)

    name = sub('(ร้อย|พัน|พล)(ตํารวจ)?(ตรี|โท|เอก)', '', name).strip()

    for t in titles:
        if name.startswith(t):
            return name[len(t):]
    return name


def vote_encoder(vote: str):
    '''
    1 = เห็นด้วย,
    2 = ไม่เห็นด้วย,
    3 = งดออกเสียง,
    4 = ไม่ลงคะแนนเสียง, 
    5 = ไม่เข้าร่วมประชุม,
    \- = ไม่ใช่วาระการประชุม
    '''
    if vote == 'เห็นด้วย':
        return 1
    if vote == 'ไม่เห็นด้วย':
        return 2
    if vote == 'งดออกเสียง':
        return 3
    if vote == 'ไม่ลงคะแนนเสียง':
        return 4
    if vote == '-':
        return 5

    p = (similar(vote, 'เห็นด้วย'), similar(vote, 'ไม่เห็นด้วย'),
         similar(vote, 'งดออกเสียง'), similar(vote, 'ไม่ลงคะแนนเสียง'))
    i = np.argmax(p)
    if p[i] < .8:
        return 5
    return i + 1


def last_name(fullname):
    sep_fn = fullname.split(' ', maxsplit=1)
    if (len(sep_fn) > 1):
        return sep_fn[-1].replace('\'', '')
    return fullname


def match_name(df, df_ocr, included_b):
    match_row = np.full(df.name.shape, np.nan)
    df_cleaned = df_ocr[included_b]

    for i, g in df.groupby('party'):
        tmp = df_cleaned[df_cleaned['party'] == i]
        for j, r in tmp.iterrows():
            sim = g.apply(lambda x: similar(r.fullname.replace(
                ' ', ''), x['name'] + x['lastname']), axis=1)
            if sim.max() > .6:
                if j not in match_row:
                    match_row[g.iloc[sim.argmax()].name] = j

    for i, row in df_ocr[included_b & (df_ocr.index.isin(np.unique(match_row)) == False)].iterrows():
        sim = df.apply(lambda x: similar(row.fullname.replace(
            ' ', ''), x['name'] + x['lastname']), axis=1)
        if sim.max() > .7:
            match_row[sim.argmax()] = i

    return match_row


if __name__ == '__main__':
    main(*argv[1:])
