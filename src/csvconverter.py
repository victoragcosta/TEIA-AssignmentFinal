# Module to handle csv operations.

# Package imports:
import csv

# Public methods:

## Method to convert training history data into the csv format:
def converter(a):
  out = []

  for i in range(0,len(list(a.values())[0])):
    el = {}
    for k,e in a.items():
      el[k] = e[i]
    out.append(el)

  return out

## Method to save the csv data in a file:
def savecsv(rows, filename='result.csv'):
  with open(filename, 'w') as f:
    w = csv.DictWriter(f, rows[0].keys())
    w.writeheader()
    for data in rows:
      w.writerow(data)
