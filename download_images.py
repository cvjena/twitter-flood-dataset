import sys
import os.path
import requests
import json
from tqdm import tqdm


def download_images(imgs, directory):
    
    num_dl = 0
    for img in tqdm(imgs.values()):
        req = requests.get(img['URL'], timeout=10)
        if req.status_code == 200:
            with open(os.path.join(directory, os.path.basename(img['URL'])), 'wb') as f:
                for chunk in req:
                    f.write(chunk)
            num_dl += 1
        req.close()
    return num_dl


if __name__ == '__main__':

    datasets = sys.argv[1:] if len(sys.argv) > 1 else ['harz17', 'rhine18']
    for ds in datasets:
        with open('{}.json'.format(ds)) as f:
            imgs = json.load(f)
        if not os.path.exists(ds):
            os.mkdir(ds)
        print('Downloading images from {} dataset...'.format(ds))
        num_dl = download_images(imgs, ds)
        print('Successfully downloaded {} of {} images from {} dataset.'.format(num_dl, len(imgs), ds))

