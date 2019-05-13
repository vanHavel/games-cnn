import os

import steam
from network_error import NetworkError

# get app info and screens for supplied game ids
apps_path = os.path.join('raw_data', 'apps_long.txt')
counter = 0
id_file = open(apps_path, 'r')
for line in id_file.readlines():
    app_id = line[:-1]
    done = False
    attempts = 0
    print(counter)
    counter += 1
    while (not done):
        try:
            info = steam.get_app_info(app_id)
            if not info == None:
                steam.store_app_info(info)
            done = True
        except NetworkError as e:
            print(e)
            attempts += 1
            if attempts == 3:
                done = True
