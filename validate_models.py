import subprocess, string
import csv
from multitron import train
import numpy as np


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

def parse_cm( output ):
	'''
	output: multiline string output of running weka from command line
	return the confusion matrix as a 2d list
	'''
	in_cm_sec = 0
	cm = []
	for line in output.split('\n'):
		parts = line.split()

		if in_cm_sec == 3 and len(parts) > 0 and parts[0] != "a":
			cm_end = parts.index('|')
			cm += [[int(e) for e in parts[:cm_end]]]

		if 'Confusion' in parts and 'Matrix' in parts:
			in_cm_sec += 1 

		if in_cm_sec >= 2 and len(parts) == 0:
			in_cm_sec += 1 

	return cm

def run_weka(model, dataset_name):
	args = [
			'java',
			'-cp',
			'/Applications/weka-3-8-0-oracle-jvm.app/Contents/Java/weka.jar',
			'weka.classifiers.'+model,
			'-t',
			'out/filtered_{}_train.arff'.format(dataset_name),
			'-T',
			'out/filtered_{}_test.arff'.format(dataset_name)
			]
	out = get_subprocess_out( args )

	return out

def read_csv( fid ):
	data = {}
	with open(fid, 'r') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			for k in row:
				if k not in data: data[k] = []
				data[k] += [float(row[k])]

	return data

def get_averages( tabular ):
	'''
	calc averages for each col in a csv
	return a dict of { col_header : col_average, ... }
	'''

	avgs = {}
	for k in list(tabular[0]):
		avgs[k] = sum([row[k] for row in tabular])/len(tabular)
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
	print(h_str)
	print( "|".join(['---']*len(data[0])))
	for row in data:
		# format floats
		row = {k:"{:.2f}".format(row[k]) if type(row[k])==float else row[k] 
				for k in row}
		print(" | ".join([str(row[h]) for h in headers]))

def format_as_tabular( data, key_header, val_header ):
	'''
	*as tabular b/c this is the format csv module uses*
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

def format_cm_as_tabular( data ):
	'''
	format mxm confusion matrix as 
	{a:val_1,b:val_2,...}
	return tabular form
	'''
	lowers = string.ascii_lowercase
	tabular_data = []
	for d in data:
		row = {}
		for i in range(len(d)):
			row[lowers[i]] = float(d[i])
		tabular_data += [row]

	return tabular_data


def write_csv( csv_fid, data ):
	with open(csv_fid , 'w') as csvfile:
		fieldnames = list( data[0] )
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		writer.writeheader()
		for e in data:
			writer.writerow(e)

def avg_cms( cms ):
	'''
	cms: {model_name: [set of confusion matrices], ...}
	return {model_name: [average of confusion matrices], ...}
	'''
	avgs = {}
	for m in cms:
		a = np.array(cms[m])
		a_flat = a.reshape(a.shape[0],4)

		avg_flat = np.average(a_flat.T, axis=1)
		avgs[m] = avg_flat.reshape(2,2)

	return avgs

def validate(n_trials):

	accs = [] 
	cms = {}  # confusion matrices
	
	for i in range(n_trials):
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
		for m in models:
			out = run_weka(m, dataset_name)
			acc[m] = parse_acc( out )

			if m not in cms: cms[m] = []
			cms[m] += [parse_cm( out )]

		m = 'multitron'
		acc[m], cm = train( dataset_name = dataset_name )
		acc[m] *= 100
		if m not in cms: cms[m] = []
		cms[m] += [cm]

		accs += [acc]

		print(acc)

	csv_fid = 'out/trials_{}.csv'.format( dataset_name )
	# WRITE to csv
	write_csv( csv_fid, accs )

	return accs, cms

if __name__ == '__main__':
	accs, cms = validate(50)

	# SHOW CMs
	acms = avg_cms( cms )
	for m in acms:
		print('\n'+m)
		display_md_table(format_cm_as_tabular(acms[m]))

	avgs = get_averages( accs )
	table_data = format_as_tabular( avgs, 'model', 'accuracy' ) 
	display_md_table( table_data )
