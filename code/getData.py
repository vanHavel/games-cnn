

import steam

id_file = open('apps_long.txt', 'r')
for line in id_file.readlines()[8798:]:
    app_id = line[:-1]
    print(app_id)
    done = False
    while (not done):
        try:
            info = steam.get_app_info(app_id)
            if not info == None:
                steam.store_app_info(info)
            done = True
        except IOError as e:
            print(e.strerror)
