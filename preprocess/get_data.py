import os

import steam

apps_path = os.path.join('raw_data', 'apps_long.txt')

id_file = open(apps_path, 'r')
for line in id_file.readlines()[8798:]:
    app_id = line[:-1]
    done = False
    while (not done):
        try:
            info = steam.get_app_info(app_id)
            if not info == None:
                steam.store_app_info(info)
            done = True
        except IOError as e:
            print(e.strerror)
