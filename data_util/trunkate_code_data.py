import csv

data = []
with open('code_data.csv','r') as csvfile:
	reader = csv.DictReader(csvfile)
	for row in reader:
		row['code'] = ""
		data += [row]

with open('code_data.trunk.csv','w') as csvfile:
	writer = csv.DictWriter(csvfile, fieldnames=list(data[0]))
	for row in data:
		row = { k:row[k] for k in row if k != None}
		writer.writerow(row)
