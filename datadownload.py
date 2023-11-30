import wget

# simple code for downloading fullbody CT scans from Visible human Korean
prefix = 'https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Male-Images/Fullcolor/head/'
path = 'C://Users/Abdelhak/Documents\DATASETS/Visible Human/Korean VHD Full Body/FullColor_head/'

for i in range(1001, 1378):
    filename = f'a_vm{i}.raw.Z'
    url = prefix + filename
    file = path + filename
    wget.download(url, file, bar=wget.bar_thermometer)
