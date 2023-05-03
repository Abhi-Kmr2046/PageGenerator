import xml.etree.ElementTree as ET
import os

map_file = open("mapping.csv", "w")

def parse(file):
    tree = ET.parse(file)
    root = tree.getroot()
    # print(root.findall('handwritten-part'))
    for item in root.findall('./handwritten-part/line/word'):
        file_name = item.attrib['id']
        sp = file_name.split('-')
        # print(file_name, sp)
        path = f"C:/Users/abhis/Documents/BTP/words/{sp[0]}/{sp[0]}-{sp[1]}/{file_name}.png"
        # print(path)
        if not os.path.exists(path):
            print(f'{path} does not exist')
            break
        print(f"{item.attrib['text']}\t{path}", file=map_file)



directory = r"C:\Users\abhis\Documents\BTP\xml"
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        parse(f)

