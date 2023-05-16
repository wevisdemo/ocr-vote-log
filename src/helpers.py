from typing import List, Union
import cv2
from pdf2image import convert_from_bytes, convert_from_path
import pandas as pd
import numpy as np
import re
from tqdm.notebook import tqdm
from requests import get as req_get
from difflib import SequenceMatcher
from functools import partial
import requests
import os
import json
from redis import Redis

def get_image_from_path(path: str) -> List[np.array]:
    """
    Gets an image from a path.

    Args:
        path: The path to the image.

    Returns:
        A list of numpy arrays, one for each page in the image.

    Raises:
        FileNotFoundError: If the path does not exist.
        TypeError: If the path is not a string.
    """

    # Check if the path is a string.
    if not isinstance(path, str):
        raise TypeError("path must be a string")

    # If path is a local file
    if os.path.isfile(path):
        # Convert local file into a list of numpy arrays
        converted = convert_from_path(path)

    # If path is a URL
    elif re.match(r'^https?://', path) or re.match(r'^www\.', path):
        # Fetch data from URL
        response = req_get(path)

        # Convert fetched data into a list of numpy arrays
        converted = convert_from_bytes(response.content)

    pages = [np.array(im) for im in converted]
    return pages


def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)


def detect_column(images):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    hist_avg = None
    for im in images:
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        noise_removed = noise_removal(gray)
        th, threshed = cv2.threshold(
            noise_removed, 200, 255, cv2.THRESH_BINARY_INV)
        dilated = cv2.dilate(threshed, kernel, iterations=3)

        if hist_avg is None:
            hist_avg = cv2.reduce(dilated, 0, cv2.REDUCE_AVG)
        else:
            hist_avg = (hist_avg + cv2.reduce(dilated, 0, cv2.REDUCE_AVG)) / 2

    hist_avg = hist_avg.reshape(-1)

    hist_b = hist_avg < hist_avg.max() - hist_avg.std()
    cols1 = np.argwhere(hist_b[1:] & (hist_b[1:] ^ hist_b[:-1])).reshape(-1)
    cols2 = np.argwhere(hist_b[:-1] & (hist_b[1:] ^ hist_b[:-1])).reshape(-1)
    cols = (cols1 + ((cols1 - cols2) * 0.8)).astype(int)

    return cols


def draw_img_boxes(image, boxes):
    # creat black image
    local_img = np.zeros(image.shape)
    for x, y, w, h in boxes:
        # draw text bounding box
        local_img[y:y+h, x:x+w] = 255
    return local_img


def parse_text(page_list, reader, columns, log_writer=None):
    rects_in_lines = []
    log = []
    for i, image in enumerate(tqdm(page_list)):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _log_data = dict(list=list(), page_name=i)

        blured = cv2.GaussianBlur(gray, (9, 9), 0)
        th, threshed = cv2.threshold(
            blured, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        threshed = cv2.dilate(threshed, kernel)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        dilated = cv2.dilate(threshed, kernel)
        contours, hier = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rects = [cv2.boundingRect(c) for c in contours]

        h, w = image.shape[:2]

        # filter border
        rects = [rect for rect in rects
                 if rect[0] and rect[1]
                 and w-rect[0]+rect[2]
                 and h-rect[1]+rect[3]
                 and rect[2]*rect[3] > 200]

        rects.sort(key=lambda rect: rect[1])

        b_image = draw_img_boxes(gray, rects)
        # cv2.imwrite('empty_img.png', b_image)

        hist = cv2.reduce(b_image, 1, cv2.REDUCE_AVG).reshape(-1)

        h, w, _ = image.shape
        stack_y = []
        for ii in range(h-1):
            if (hist[ii] and not hist[ii+1]):
                stack_y.append(ii)
            elif (not hist[ii] and hist[ii+1]):
                stack_y.append(ii)

        # draw_img = image.copy()
        # for x, y, w, h in rects:
        #   draw_img = cv2.rectangle(draw_img, (x,y),(x+w,y+h), 255, 1)
        # cv2.imwrite('rect-img-{:02}.png'.format(i), draw_img)

        # draw_img = image.copy()
        while(stack_y):
            rects_in_col = [[] for _ in range(len(columns)+1)]
            line_y2 = stack_y.pop(0)
            if stack_y:
                line_y1 = stack_y.pop(0)
            else:
                line_y1 = image.shape[0]
            line_rects = []
            while rects:
                rect = rects[0]
                y_mid_point = rect[1] + rect[3]//2
                x_mid_point = rect[0] + rect[2]//2
                if line_y2 < y_mid_point < line_y1:
                    rect = rects.pop(0)
                    line_rects.append(rect)
                else:
                    break
            line_rects.sort(key=lambda x: x[0])

            for x, y, w, h in line_rects:
                col = (x + (w/2) > columns).sum()
                # draw_img = cv2.rectangle(draw_img, (x,y),(x+w,y+h), 255, 1)
                text = reader.recognize(image[y:y+h, x:x+w])[0][1]
                _log_data['list'].append(dict(text=text,
                                              bbox=(x, y, w, h),
                                              col=col,
                                              line=len(rects_in_lines),))
                if col >= len(rects_in_col):
                    log.append((col, text))
                    continue
                rects_in_col[col].append(text)
            rects_in_lines.append(rects_in_col)
        if log_writer:
            log_writer.write_log('parse_text', _log_data)
        # cv2.imwrite('ocr-img-{:02}.png'.format(i),draw_img)
    if log:
        print(log)
    return rects_in_lines


def padding(images: np.array):
    """
    A function that takes an array of images and
    returns a new array with all the images padded to have the same shape.
    """

    shapes: List = []  # A list to hold the shapes of all the images in the input array
    for image in images:
        shapes.append(image.shape)
    # Calculate the maximum shape across all images
    max_shape = np.max(shapes, 0)

    padded: List = list()  # A list to hold the padded images
    for image in images:
        # Calculate the amount of padding needed in each dimension
        y, x, z = (max_shape - image.shape)
        # Add the padding to the image using the OpenCV `copyMakeBorder` function
        p = cv2.copyMakeBorder(
            image, 0, y, 0, x, cv2.BORDER_CONSTANT, None, (255, 255, 255))
        padded.append(p)  # Add the padded image to the output list

    return padded  # Return the list of padded images


def find_interesting_rows(df):
    name_column = 2
    party_column = 3

    return ((
        df[0].apply(lambda x: re.sub('[\^:\.\s]', '', str(x)).isdigit())
        | df[1].apply(lambda x: re.sub('[\^:\.\s]', '', str(x)).isdigit())
        | df[party_column].apply(lambda x: str(x).startswith('พรรค'))
    ) & (~df[name_column].isna()))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class VoteLog:
    def __init__(self, pdf_url: str, logger=None):
        self.pdf_url = pdf_url
        self.vote_id = self._get_vote_log_id(pdf_url)
        self.data_table = None
        self.col2type = {}
        self.type2col = {}
        self.logger = logger

    def write_log(self, ftype: str, data: Union[list, dict]):
        assert isinstance(ftype, str)
        if self.logger is None:
            raise ValueError('Logger not defined')
        else:
            radis: Redis = self.logger

        if not isinstance(data, str):
            data = json.dumps(data, cls=NpEncoder, ensure_ascii=False)

        radis.hsetnx('votes', self.pdf_url, self.vote_id)
        if ftype == 'parse_text':
            radis.lpush(self.vote_id, data)
        elif ftype == 'result':
            radis.hset('result_table', self.vote_id, data)
        else:
            raise ValueError('argrument `ftype` not recognized')

    def summary(self,):
        vote_column = self.type2col['vote']
        print('สรุปการลงคะแนน (ตรวจความถูกต้องจากหัวเอกสาร)')
        for v, c in self.data_table[vote_column].value_counts().items():
            print(f'\t{v}\t{c}')

    def _get_vote_log_id(self, url):
        return re.sub(r'.*/(.*)\.pdf$', r'\1', url)

    def set_data_table(self, data_frame: pd.DataFrame):
        self.data_table = data_frame
        self._build_vote_column_type()

    def put_vote(self, people_df, people_index, ocr_index):
        vote_column = self.type2col['vote']
        people_df.loc[people_index,
                      self.vote_id] = self.data_table.loc[ocr_index][vote_column]
        self.data_table.loc[ocr_index, 'people_id'] = people_index

    def _build_vote_column_type(self,):
        self.col2type = {}
        for idx, column in self.data_table.astype(str).iteritems():
            for row in column:
                if row.startswith('พรรค') or 'สมาชิกวุฒิสภา' in row:
                    self.col2type[idx] = 'party'
                elif row.startswith('นาย') or row.startswith('นาง'):
                    self.col2type[idx] = 'name'
                elif 'เห็นด้วย' in row or 'ไม่เห็นด้วย' in row or 'ลงคะแนนเสียง' in row:
                    self.col2type[idx] = 'vote'
                if idx in self.col2type.keys():
                    break
        self.type2col = {t: c for c, t in self.col2type.items()}

    def _fix_vote_replacer(self, text: str):
        return text.replace('.ท็น', 'เห็น').replace('เท็น', 'เห็น').replace('เห็นด้าย', 'เห็นด้วย')

    def fix_vote(self, vote: str):
        vote_types = ['เห็นด้วย', 'ไม่เห็นด้วย',
                      'งดออกเสียง', 'ไม่ลงคะแนนเสียง', '-']
        if not isinstance(vote, str):
            return '-'
        # common ocr error
        vote = self._fix_vote_replacer(vote)

        if vote in vote_types:
            return vote
        p = [similar(vote, v) for v in vote_types]
        i = np.argmax(p)
        if p[i] < .6:
            return '-'
        return vote_types[i]

    def fix_votes(self,):
        vote_column = self.type2col['vote']
        self.data_table[vote_column] = self.data_table[vote_column].apply(
            self.fix_vote)

    def export(self, people_df):
        # rename columns
        temp = self.data_table.rename(columns=self.col2type)
        temp = temp.join(people_df.assign(
            people_id=people_df.index), 'people_id', lsuffix='_ocr')
        temp.loc[:, 'vote_id'] = self.vote_id
        temp.loc[:, 'people_id'] = temp['people_id_ocr']
        return temp[['vote_id', 'name_ocr', 'name', 'people_id', 'vote']]


def get_data_frame(vote_log: VoteLog, reader,):
    # Download PDF and convert to numpy array
    pages = get_image_from_path(vote_log.pdf_url)

    # Pad images to have the same dimensions
    pages = padding(pages)

    # Find column in the document
    columns = detect_column(pages)

    # Parse text
    rects_in_lines = parse_text(pages, reader, columns, vote_log)

    # Concatenate string in table cell
    df = pd.DataFrame([[' '.join(r) for r in line] for line in rects_in_lines])

    # Save DataFrame to CSV file, for debugging
    df.to_csv(f'{vote_log.vote_id}.csv', index=False)

    # Find interesting rows
    interested_row_b = find_interesting_rows(df)

    # Create a Boolean mask that indicates whether any row contains the word "หมายเหตุ".
    is_note_b = df[0].apply(lambda x: isinstance(x, str) and 'หมายเหตุ' in x)

    # Check if any of the rows contain the word "หมายเหตุ".
    has_note = is_note_b.any()

    print('ในบันทึกการออกคะแนนนี้', 'มี' if has_note else 'ไม่มี', 'หมายเหตุ')

    # If any of the rows contain the word "หมายเหตุ",
    # set the corresponding values in the `interested_row_b` mask to False.
    if has_note:
        interested_row_b.loc[df[is_note_b].index[0]:] = False

    print(
        f'มีรายชื่อทั้งหมด {interested_row_b.sum()} รายชื่อ (ไม่นับหลังหมายเหตุ)')

    # Filter DataFrame
    df = df[interested_row_b]

    vote_log.set_data_table(df)


def get_data(url, params={}):
    """
    A function that receives a url and optional parameters,
    and returns a list of data obtained from that url.
    """

    page_size = 100  # The number of data items to retrieve in each request
    page_idx = 0   # The index of the current page
    has_next = True  # A flag indicating whether there is more data to retrieve
    data_list = []  # The list that will hold the retrieved data

    # The loop that retrieves the data in pages of size `page_size`
    while has_next:
        # Update the parameters with the current offset and limit values
        params.update(dict(limit=page_size, offset=page_idx*page_size,))

        # Send a GET request to the specified URL with the updated parameters
        resp = requests.get(url, params=params)

        # Raise an exception if there was an error in the request
        resp.raise_for_status()

        # Extract the data from the response JSON
        resp_data = resp.json()['data']

        # Append the retrieved data to the list
        data_list += resp_data['list']

        # Check if there is more data to retrieve
        has_next = not resp_data['pageInfo']['isLastPage']

        # Move to the next page
        page_idx += 1

    # Return the retrieved data as a list
    return data_list


def get_members_df():
    """
    get_members_df function fetches data from an api or local json files and converts them into a pandas dataframe.
    It fetches data from two different APIs containing information about parliament members and their respective parties.
    The function then converts the fetched data into pandas dataframe and returns a joined dataframe.

    Args:
      None

    Returns:
      A pandas dataframe containing parliament members and their respective parties.
    """
    # Helper function to load data from api or local json
    def load_data_from_api_or_local(url: str, params: dict, local_json: str) -> dict:
        # Check if local json file exists
        if os.path.exists(local_json):
            # If exists, load data from local json
            data = json.load(open(local_json))
        else:
            # If not exists, fetch data from API
            data = get_data(url, params)
            # Save fetched data to local json file
            json.dump(data, open(local_json, 'w'), ensure_ascii=False)
        return data

    # Fetch data about parliament members from API or local json file
    parl_mems = load_data_from_api_or_local(
        'https://sheets.wevis.info/api/v1/db/public/shared-view/572c5e5c-a3d8-440f-9a70-3c4c773543ec/rows',
        dict(fields='Id,Name,IsMp,IsSenator'),  # where='(IsActive,is,true)'
        'parliament_members_table.json')
    # Fetch data about parliament members and their respective parties from API or local json file
    parl_mems_parties = load_data_from_api_or_local(
        'https://sheets.wevis.info/api/v1/db/public/shared-view/707598ab-a5db-4c46-886c-f59934c9936b/rows',
        dict(fields='Person,Party',),
        'parliament_member_party_table.json')

    # Helper function to convert data into dataframe
    def member_party_converter(p: dict) -> dict:
        return dict(party=p['Party'] if not p['Party'] else p['Party']['Name'],
                    party_id=p['Party'] if not p['Party'] else p['Party']['Id'],
                    people_id=p['Person']['Id'],)

    def member_converter(m: dict) -> dict:
        return dict(id=m['Id'], name=m['Name'], is_mp=m['IsMp'], is_senator=m['IsSenator'],)

    # Convert data into pandas dataframe
    member_party: pd.DataFrame = pd.DataFrame(
        [member_party_converter(p) for p in parl_mems_parties if p['Party']])
    member_party.drop_duplicates('people_id',  keep='last', inplace=True)
    member = pd.DataFrame([member_converter(m) for m in parl_mems])

    # Set idex and join the two dataframes
    return member.set_index('id',).join(member_party.set_index('people_id'))


def similar(a: str, b: str) -> float:
    """
    similar function calculates the similarity between two strings using the SequenceMatcher class from the difflib module.
    It removes spaces in both strings before calculating the similarity ratio.

    Args:
        a: A string representing the first string to be compared.
        b: A string representing the second string to be compared.

    Returns:
        A float representing the similarity ratio between the two strings.
    """
    return SequenceMatcher(lambda x: x in ' ', a, b).ratio()


def fix_party_name(p: str, ps: List[str]) -> str:
    """
    fix_party_name function tries to fix the name of a political party by comparing it to a list of known party names.

    Args:
      p: A string representing the name of the political party to be fixed.
      ps: A list of strings representing the known party names.

    Returns:
      A string representing the fixed party name.
    """
    # If party name is empty or already in the list of known party names, return the input name
    if not p or p in ps:
        return p
    # Calculate the similarity between the input party name and the known party names
    sim_l = [similar(p, str(x)) for x in ps]
    # If the maximum similarity score is above 0.7, return the known party name with the highest similarity score
    if max(sim_l) > 0.7:
        return ps[np.argmax(sim_l).reshape(-1)[0]]
    # Otherwise, return the original party name
    return p


def correct_party_df(vote_log: VoteLog, people_df):
    """
    correct_party_df function corrects party names in the vote_log data table by comparing them with party names in the people_df dataframe.
    It creates a new column with -1 and renames it to the vote_id.
    It also creates a new column called people_index in the vote_log data table with all values set to -1.

    Args:
      vote_log: A VoteLog object representing the vote log data.
      people_df: A pandas DataFrame representing the people data.

    Returns:
      A pandas DataFrame representing the updated people data.
    """

    # Define the fix_party_name function with the party names in people_df as parameter
    fps = partial(fix_party_name, ps=people_df.party.unique())
    # Get the party column name from the vote_log object
    party_column = vote_log.type2col['party']
    # Apply the fix_party_name function to the party column of the vote_log data table
    vote_log.data_table.loc[:, party_column] = vote_log.data_table[party_column].apply(
        fps)

    # Create a new column called new_vote with -1 as initial value, rename it to vote_id and assign to people_df
    people_df = people_df.assign(
        new_vote=-1).rename(columns={'new_vote': vote_log.vote_id})
    # Create a new column called people_index with -1 as initial value and assign to vote_log data table
    vote_log.data_table = vote_log.data_table.assign(people_id=-1)

    # Apply the fix_party_name function to the party column of the vote_log data table with the party names in people_df as parameter
    vote_log.data_table.loc[:, party_column] = vote_log.data_table[party_column].apply(
        lambda x: fix_party_name(x, people_df.party.unique()))

    # Return the updated people data
    return people_df


def name_similar(ocr_name, row):
    thai_char_re = r'[^\u0E00-\u0E7F]+'
    name = re.sub(thai_char_re, '', row['name'])
    return similar(name, ocr_name)


def re_match(vote_log: VoteLog, not_matched, filtered, people_df):
    for ocr_index, ocr_name in not_matched:
        f = partial(name_similar, ocr_name)
        sim_sr = filtered.apply(f, axis=1)
        if sim_sr.max() < .5:
            continue
        people_index = filtered.iloc[sim_sr.argmax()].name
        vote_log.put_vote(people_df, people_index, ocr_index)


def match_group_by_party(vote_log: VoteLog, people_df):
    title_re = r'(นาย|นาง(สาว)?|ศาสตราจารย์|(พล|พัน)?(ตำรวจ|นาวาอากาศ)?(เอก|โท|ตรี)?)'
    non_thai_char_re = r'[^\u0E00-\u0E7F]+'
    not_matched = []

    party_column = vote_log.type2col['party']
    name_column = vote_log.type2col['name']

    def clean_name_string_fn(x):
        return re.sub(title_re, '', re.sub(non_thai_char_re, '', x))
    
    # group by party
    for party, g in vote_log.data_table.groupby(party_column):
        ocr_names = g[name_column].apply(clean_name_string_fn)

        # If party is Senator, filter people_df by is_senator column
        if party == 'สมาชิกวุฒิสภา':
            filtered_people_df = people_df[people_df.is_senator]
        else:
            filtered_people_df = people_df[(people_df.party == party)]

        for j, ocr_name in ocr_names.items():
            found = False
            index_l = []
            sim_l = []
            for i, row in filtered_people_df.iterrows():
                member_has_vote = people_df.loc[i, vote_log.vote_id] != -1
                if isinstance(member_has_vote, bool):
                    if member_has_vote:
                        continue
                elif member_has_vote.all():
                    continue
                name = re.sub(non_thai_char_re, '', row['name'])
                if name == ocr_name or name in ocr_name:
                    found = True
                    vote_log.put_vote(people_df, i, j)
                    break
                index_l.append(i)
                sim_l.append(similar(name, ocr_name))
            if not found and sim_l and max(sim_l) > .7:
                index_of_max = sim_l.index(max(sim_l))
                people_index_max = index_l[index_of_max]
                vote_log.put_vote(people_df, people_index_max, j)
            elif not found:
                not_matched.append((j, ocr_name))

    return not_matched


def matching(vote_log: VoteLog, people_df: pd.DataFrame):
    # matching
    not_matched = match_group_by_party(vote_log, people_df)
    print('👀 ชื่อที่ยังไม่พบในชีท กำลังรันอีกครั้ง',
          not_matched[:3], f'... และอีก {len(not_matched)-3}')
    # not found yet
    not_found_b = people_df[(people_df[vote_log.vote_id] == -1)]
    # matching without groupby party
    re_match(vote_log, not_matched, not_found_b, people_df)
