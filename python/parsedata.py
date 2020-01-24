import csv
from matplotlib import pyplot as plt

data    = dict()
header  = []

with open('data.csv') as file:
    reader = csv.reader(file,delimiter=',')
    for lineNumber, line in enumerate(reader):

        if lineNumber is 0:
            # Parse header
            for title in line:
                data[title] = []
            header = tuple(data.keys())
        else:
            # Parse data
            for index, value in enumerate(line):
                try:
                    value = float(value)
                except:
                    pass
                data[header[index]].append(value)

print(data.keys())

plt.plot( data['Item_Weight'], data['Item_maxDeform'], '.' )
plt.show()