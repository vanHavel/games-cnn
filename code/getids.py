import os

import steam

base_url = 'http://store.steampowered.com/search/?category1=998&page='
ids = []
for i in range(1, 616):
    ids += steam.get_app_ids_from_url(base_url + str(i))
with open('apps_long.txt', 'w') as handler:
    for app_id in ids:
        handler.write(str(app_id) + os.linesep)
