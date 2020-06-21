import pandas as pd
import re

def readJson(fileLoc):
    raw_data = pd.read_json(fileLoc)
    
    dataList = []

    for index, row in raw_data.iterrows():
        text = re.sub(r"http\S+", "", row[0])
        dataList.append(text)

    # print(dataList)
    return dataList

if __name__ == '__main__':
    readJson('data.json')