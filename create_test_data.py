from os import listdir, makedirs
import re
import fitz
import pandas as pd
import requests
import os
from time import sleep

vote_count_fpath = 'test/[WeVis] They Work for Us - Politician Data - [T] PeopleVote.csv'
vote_fpath = 'test/[WeVis] They Work for Us - Politician Data - [T] Votelog.csv'


vote_count_df = pd.read_csv(vote_count_fpath)
vote_count_df.columns = vote_count_df.iloc[0]
vote_count_df = vote_count_df.iloc[1:]

vote_df = pd.read_csv(vote_fpath)


def get_filename(url):
  return re.sub('^.*/', '', url)

def vote_of(vote_id):
  mask = vote_count_df[vote_id] != '-'
  cols = ['id', 'name', 'lastname', 'party', vote_id]
  return vote_count_df[cols][mask]

def download_pdf(url, save_dir):
  fpath = os.path.join(save_dir, get_filename(url))
  if os.path.isfile(fpath): 
    return fpath
  response = requests.get(url)
  sleep(200)
  if response.status_code != 200:
    return
  with open(fpath, 'wb') as fp:
    fp.write(response.content)
  return fpath

if __name__ == '__main__':
  root = 'test/pdf'
  vote_id_col = (vote_df.columns[2])

  urls = vote_df[[vote_id_col, 'ลิงค์ไปที่เอกสาร 1']].values
  urls = []
  for id, url in urls:
    if isinstance(id, float) or not url.endswith('.pdf'): continue
    print('inprocess...', url)
    if id in vote_count_df.columns:
      fpath = download_pdf(url, root)
      if fpath is None: continue
      csv_fpath = re.sub('.pdf$', '.csv', fpath)
      vote_of(id).to_csv(csv_fpath, index=0)
  print('done')

  zoom_mat = fitz.Matrix(2.5, 2.5)
  makedirs('test/imgs', exist_ok=True)
  for file in listdir(root):
    if not file.endswith('.pdf'):
      continue
    doc = fitz.open(root+'/'+file)
    fid = re.sub('\..*$', '', file)
    for page in doc:
      pix = page.get_pixmap(matrix=zoom_mat)
      pix.save(f"test/imgs/{fid}-{page.number:02}.png")

