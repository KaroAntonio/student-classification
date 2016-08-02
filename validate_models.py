import subprocess
import csv
from multitron import train


def get_subprocess_out( args ):
	'''
	return the ouput of the shell call defined by args
	'''
	p = subprocess.Popen(args, stdout=subprocess.PIPE, 
			 stderr=subprocess.PIPE)
	out, err = p.communicate()
	return out

def get_dataset_name():
	'''
	this function wouldnt be necesary if python had a one version policy
	'''
	args = ['python3', '-c', 'from data_util.processor import get_dataset_name as gdn; print(gdn());']
	return get_subprocess_out( args ).strip()


def parse_acc( output ):
	'''
	output: multiline string output of running weka from command line
	return accuracy value in string
	'''
	in_test_sec = 0
	for line in output.split('\n'):
		parts = line.split()
		if 'Error' in parts and 'test' in parts:
			in_test_sec = 1 

		if in_test_sec and len(parts) > 0 and parts[0] == "Correctly":
			return float(parts[4])

def run_weka(model, dataset_name):
	args = [
			'java',
			'-cp',
			'/Applications/weka-3-8-0-oracle-jvm.app/Contents/Java/weka.jar',
			'weka.classifiers.'+model,
			'-t',
			'out/filtered_{}.arff'.format(dataset_name),
			'-split-percentage',
			'70'
			]
	out = get_subprocess_out( args )

	return parse_acc( out )

def get_averages( fid ):
	'''
	calc averages for reach col in a csv
	fid: csv file id
	return a dict of { col_header : col_average, ... }
	'''
	totals = {}
	with open(fid, 'r') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			for k in row:
				if k not in totals: totals[k] = []
				totals[k] += [float(row[k])]

	avgs = {}
	for k in totals:
		avgs[k] = sum(totals[k])/len(totals[k])
	return avgs

def display_md_table( data, headers=None ):
	'''
	build a formatted table according to .md specs
	data: list of dicts
	'''
	# HEADERS
	if headers == None:
		headers = list(data[0])
	h_str = " | ".join(headers)
	print (h_str)
	print( "|".join(['---']*len(data[0])))
	for row in data:
		# format floats
		row = {k:"{:.2f}".format(row[k]) if type(row[k])==float else row[k] 
				for k in row}
		print(" | ".join([str(row[h]) for h in headers]))


def validate():
	trials = [] 

	for i in range(50):
		# Generate new data subset
		args = ['python3','convert_data.py']
		get_subprocess_out( args )

		# Validate Models (percentage-split: 70%)
		models = [
				'trees.RandomForest',
				'trees.J48',
				'trees.DecisionStump',
				'bayes.NaiveBayes',
				'bayes.BayesNet',
				'rules.DecisionTable',
				'rules.PART'
				]
		acc = {}
		args = ['python3', '-c', '"from data_util.processor import get_dataset_name as gdn; print(gdn())"']
		dataset_name = get_dataset_name()
		print('dataset',dataset_name)
		for m in models:
			acc[m] = run_weka(m, dataset_name)

		acc['multitron'] = train( dataset_name = dataset_name )*100
		print(acc)
		trials += [acc]

		csv_fid = 'out/trials_{}.csv'.format( dataset_name )
		# WRITE to csv
		with open(csv_fid , 'w') as csvfile:
			fieldnames = list( trials[0] )
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

			writer.writeheader()
			for trial in trials:
				writer.writerow(trial)

		# AVERAGE
		avgs = get_averages(csv_fid)

def format_as_tabular( data, key_header, val_header ):
	'''
	translate a dict of k:v pairs to a list of dicts with k,v as vals
	data: in the form { key: val, ... }
	return in the formL: [{key_header:key, val_header: val}, ]
	'''
	table_data = []
	for a in data:
		row = {}
		row[key_header]=a
		row[val_header]=data[a]
		table_data+=[row]
	return table_data

if __name__ == '__main__':
	validate()

	dataset_name = get_dataset_name()
	csv_fid = 'out/trials_{}.csv'.format( dataset_name )
	avgs = get_averages(csv_fid)
	table_data = format_as_tabular( avgs, 'model', 'accuracy' ) 
	display_md_table(table_data)
