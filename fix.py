import json
for enc in ['utf-8', 'cp1252', 'utf-16-le', 'utf-16']:
    try:
        data = json.load(open('Heart_Disease_BN_Project.ipynb', 'r', encoding=enc))
        print('Loaded with', enc)
        break
    except:
        pass
if 'data' in locals():
    json.dump(data, open('Heart_Disease_BN_Project.ipynb', 'w', encoding='utf-8'), indent=1)
    print('Fixed')
else:
    print('Failed')