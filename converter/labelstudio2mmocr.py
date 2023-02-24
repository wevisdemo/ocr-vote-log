import json
import argparse
import re
import os

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('filename')
  parser.add_argument('-s', '--split_train_test', help='split train test or not', action='store_true',)
  parser.add_argument('-m', '--move_image', action='store_true', default=False)
  parser.add_argument('-d', '--directory_name',)

  args = parser.parse_args()

  with open(args.filename) as fp:
    ls_data = json.load(fp)

  # create output directory
  if args.directory_name is not None:
    output_dir_name = args.directory_name
  else:
    output_dir_name = re.sub(r'([^/]+)\.json', r'\1', args.filename)
  if not os.path.exists(output_dir_name):
    os.mkdir(output_dir_name)

  # create directory for images
  output_image_dir_name = os.path.join(output_dir_name, 'imgs')
  if not os.path.exists(output_image_dir_name):
    os.mkdir(output_image_dir_name)
  
  # prepare output data shape
  mm_data_list = list()
  output_data = dict(
    metainfo=dict(
      dataset_type='TextDetDataset',
      task_name='textdet',
      category=[dict(id=0, name='text')]
    ),
  )
  
  for file in ls_data:
    label_instances = list()
    
    for label in file['label']:
      if label['rectanglelabels'][0] != 'text':
        continue

      img_w, img_h = label['original_width'], label['original_height']
      x, y, w, h = label['x']/100, label['y']/100, label['width']/100, label['height']/100
      x0, y0, x1, y1 = x*img_w, y*img_h, (x+w)*img_w, (y+h)*img_h
      label_instances.append(dict(
        polygon=[x0, y0, x1, y0, x1, y1, x0, y1],
        bbox=[x0, y0, x1, y1],
        bbox_label=0,
        ignore=False
      ))

    file_data = dict(
      instances=label_instances,
      img_path=file['ocr'].rsplit('/',maxsplit=1)[-1],
      height=file['label'][0]['original_height'],
      width=file['label'][0]['original_width'],
      seg_map='gt_'+re.sub('[^\.]+$','txt',file['ocr'].rsplit('/',maxsplit=1)[-1])
    )

    # copy file from original
    if args.move_image:
      os.system('cp {} {}'.format(file['ocr'].split('?d=')[1], output_image_dir_name))

    mm_data_list.append(file_data)
  
  if args.split_train_test:
    train_size = int(len(mm_data_list) * .8)

    with open(os.path.join(output_dir_name, 'textdet_train.json'), 'w') as fp:
      output_data['data_list'] = mm_data_list[:train_size]
      json.dump(output_data, fp,)

    with open(os.path.join(output_dir_name, 'textdet_test.json'), 'w') as fp:
      output_data['data_list'] = mm_data_list[train_size:]
      json.dump(output_data, fp,)
    
  else:
    with open(os.path.join(output_dir_name, 'textdet.json'), 'w') as fp:
      output_data['data_list'] = mm_data_list
      json.dump(output_data, fp,)
