from typing import List
import requests
from pdf2image import convert_from_bytes
import numpy as np

def matrix_from_url(url) -> List[np.ndarray]:
  response = requests.get(url)
  if response.status_code != 200:
    raise ValueError(f'status code {response.status_code} (expected 200) [{url}]')
  converted = convert_from_bytes(response.content)
  pages = [np.array(im) for im in converted]
  return pages