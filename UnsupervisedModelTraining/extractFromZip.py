import zipfile

with zipfile.ZipFile("Train.zip", 'r') as zip_ref:
    zip_ref.extractall()
