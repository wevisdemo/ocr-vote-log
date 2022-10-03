import pandas as pd

vote_count_fpath = 'test/[WeVis] They Work for Us - Politician Data - [T] PeopleVote.csv'
vote_fpath = 'test/[WeVis] They Work for Us - Politician Data - [T] Votelog.csv'

if __name__ == '__main__':
  vote_count_df = pd.read_csv(vote_count_fpath)
  vote_count_df.columns = vote_count_df.iloc[0]

  vote_df = pd.read_csv(vote_fpath)
  for pdf_url in vote_df['ลิงค์ไปที่เอกสาร 1']:
    print(pdf_url)