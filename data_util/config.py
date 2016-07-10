import csv

def load_configs_from_csv(fid):
	configs = []
	with open(fid,'r') as data:
		for row in csv.DictReader(data):
			conf = {}
			for e in row:
				try: conf[e] = eval[row[e]]
				except: conf[e] = row[e]
			configs += [conf]

	return configs


def get_config():
	# central config
	data_conf = {
            'grades_fid':'data/grades.csv',
            'totals_fid':'data/totals.csv',
            'arff_fid':'out/features.arff',    # None to load from progsnap
            #'arff_fid':None,    # None to load from progsnap
            'out_dir':'out',
            'progsnap_dir':'data/progsnap_data',
            'n_assignments':None,  # None for all
			}
	
	global_conf = {
            'rehearse_an':[87,131,43,88,92], # an of rehearse assignments
            'prepare_an':[],
            'perform_an':[82,55,49,54,48,41,47,35,36,37,45,39,59,64,56,62,42,61,63,57,58,91,67],
        }

	data_conf.update(global_conf)

	return data_conf

