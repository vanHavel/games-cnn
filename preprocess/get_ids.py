import os

import steam

# get ids of games on steam
numpages = 670

base_url = 'http://store.steampowered.com/search/?category1=998&page='
ids = []
for i in range(1, numpages):
    ids += steam.get_app_ids_from_url(base_url + str(i))
app_path = os.path.join('raw_data', 'apps_long.txt')
with open(app_path, 'w') as handler:
    for app_id in ids:
        handler.write(str(app_id) + os.linesep)
