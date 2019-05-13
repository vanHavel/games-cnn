# needs requests package
import requests
import os
import re

from network_error import NetworkError

# get info for given id
def get_app_info(app_id):
    dirname = str(app_id)
    # skip id, as info already there
    if os.path.exists(dirname):
        return None
    response = requests.get('https://store.steampowered.com/api/appdetails/?appids=' + str(app_id))
    if response.ok:
        jo = response.json()[str(app_id)]
        if jo['success']:
            return jo['data']
        else:
            raise NetworkError('Request returned "success: False"')
    else:
        raise NetworkError('Request failed')

# store app info and screenshots
def store_app_info(info):
    # check that type is game and has screenshots, otherwise skip
    if not info['type'] == 'game':
        return
    if not 'screenshots' in info:
        return
    # make dir with app id if it does not exist, otherwise skip
    app_id = info['steam_appid']
    dirname = os.path.join('raw_data', str(app_id))
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    else:
        return
    # write info file
    info_filename = os.path.join(dirname, 'info.json')
    info_file = open(info_filename, 'w')
    info_file.write(str(info))
    info_file.close()
    # get maximum 5 screenshots and save them
    if 'screenshots' in info:
        for screen_info in info['screenshots'][0:5]:
            screen_id = screen_info['id']
            screen_url = screen_info['path_full']
            screen_filename = os.path.join(dirname, str(screen_id) + ".jpg")
            response = requests.get(screen_url)
            if response.ok:
                image_data = response.content
                screen_file = open(screen_filename, 'wb')
                screen_file.write(image_data)
                screen_file.close()
            else:
                raise NetworkError('Failed to get screenshot ' + str(screen_id) + ' for game ' + str(app_id))

# search for app ids in html
def find_app_ids(html):
    pattern = re.compile('https://store.steampowered.com/app/\d*')
    results = pattern.findall(html)
    ids = list(map(lambda s: s[35:], results))
    return ids

# request html and search for app ids
def get_app_ids_from_url(url):
    response = requests.get(url)
    if response.ok:
        return find_app_ids(response.text)
    else:
        raise NetworkError('could not fetch url ' + url)
