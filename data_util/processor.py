'''
Convert a data set from progsnap format
supplemented with student grade data
to a weka readable format
'''

import sys, os, random
import arff, csv, json
import re
# import traceback
import subprocess
from datetime import datetime as dt
import numpy as np

import progsnap
from helpers import *
from docopt import docopt

def get_dataset_name(fid='data_util/dataset_name.txt'):
	'''
	get the dataset name from wherever she happens to be
	'''
	with open( fid ,'r') as f:
		return f.readline().strip()


class DataProcessor(object):

	def __init__(self, config={} ):

		# Load Config
		self.load_config( config )

		# assigment numbers
		# TODO move this to an external file
		self.prepare_an = []
		self.rehearse_an = [87,131,43,88,92]
		self.perform_an = [82,55,49,54,48,41,47,35,36,37,45,39,59,64,56,62,42,61,63,57,58,91,67]
		self.common_an = [41, 47, 48, 49, 50, 52, 54, 55, 82, 83, 84] # for both 2016 and 201509
		
		self.students = {}
		self.train = []
		self.test = []
		self.an_ts = {}  # assignment number timestamps 
		self.an_cutoffs = {}
		self.la_labels = [] # TODO make consistent la -> an
		self.out_suff = []
		self.classes = {} # keep track of classifications

		self.batch_ptr = 0
		self.null_int = -1

		if self.arff_fid:
			self.load_arff()
		elif self.progsnap_dir:
			self.load_progsnap()
			self.load_grades()
		elif self.csv_fid:
			self.load_csv()
		else:
			raise Exception('Specify load source: arff_fid/csv_fid/progsnap_dir')

		self.process()

	def load_config(self, config ):
		'''
		LOAD data config
		use defaults as needed
		'''
		if 'dataset_name' in config:
			dataset_name = config['dataset_name']
		else:
			dataset_name = get_dataset_name()
		# TODO ... is this a good place to put this?
		defaults = {
			'dataset_name':dataset_name,
            'grades_fid':'data/'+dataset_name+'/grades.csv',
            'totals_fid':'data/'+dataset_name+'/totals.csv',
            'csv_fid':None,
            'arff_fid': None,   # None to load from progsnap
            'progsnap_dir':None,   #
            'out_dir':'out',
            'n_assignments':None,  # None for all
			'verbose':True,
			'nominal':True,
            }
		
		# replace defaults with given config
		for k in config:
			if k not in defaults:
				raise Exception("{} not valid config.".format(k))
			defaults[k] = config[k]
		
		# instantiate class params
		for k in defaults:
			val = defaults[k]
			val = '"{}"'.format(val) if type(val) == str else val
			exec('self.{} = {}'.format(k,val))

	def load_csv(self):
		'''
		for loading from the cleaned up ahadi data csv file
		'''
		if self.verbose: 
			print("Loading from csv: {}".format(self.csv_fid))
		with open(self.csv_fid,'r') as csvfile:
			reader = csv.DictReader(csvfile)
			
			for row in reader:
				# set ts for assignments
				sn = row['id']
				del row['id']
				for k in row:
					try: row[k] = int(row[k])
					except: pass
					try: row[k] = float(row[k])
					except: pass
				self.students[sn] = row

	def load_arff(self):
		if self.verbose: 
			print("Loading from arff")
		for row in list(arff.load(self.arff_fid)):
			stats = row._data
			sn = stats['student_num']

			# apparently the arff module has a bunch of int keys that i dont need?
			# remove them
			bad_keys = ['student_num']
			for k in stats:
				if type(k)==type(0):
					bad_keys+=[k]
			for bk in bad_keys:
				del stats[bk]
			self.students[sn] = stats

	def disp_feat_stats(self, feat ):
		feats = self.compile_regexes([feat])
		print("{} stats".format(feats))
		stats = self.feats_stats(feats)
		for stat in stats:
			print("\t{}: {:.2f}".format(stat,stats[stat]))

	def feats_stats(self, feats ):
		''''
		feats: a list of feats
		return a dict with stats for feature, feature may be a regex
		avg, mean, std dev
		'''
		a = []	
		for feat in feats:
			a += [np.array([self.students[sn][feat] for sn in self.students])]
		a = np.array(a)
		return {
				'mu':a.mean(),
				'sigma':a.std(),
				'median':np.median(a)
				}

	def feature_dist(self, feature ):
		'''
		return the distribution for all values of a feature asa a dict
		'''
		counts = self.feature_count( feature )
		total = float(sum(counts.values()))
		return {k:v/total for k,v in counts.items()}

	def n_features( self ):
		'''
		return the number of features in the dataset
		'''
		sn0 = list(self.students)[0]
		return len(self.students[sn0])

	def feature_count(self, feature ):
		'''
		return the count for all values of a feauture as a dict
		dist = {val_1: count_1, ...}
		'''
		dist = {}
		for sn in self.students:
			s = self.students[sn]
			if feature not in s:
				raise Exception('{} not present in all students'.format(feature))
			val = s[feature]
			if val not in dist: dist[val] = 0 
			dist[val] += 1

		return dist

	def compile_regexes( self, regexes ):
		# if given a list, return a list of attributes matching the regexes
		# if given a dict, return a dict with attrs as keys matching the regexes
		students = self.students
		sn_0 = list(students.keys())[0]
		curr_attr = list(students[sn_0].keys())

		# compile regexes
		is_list = type( regexes ) == list
		compiled = [] if is_list else {}
		for r in regexes:
			for attr in curr_attr:
				if re.match(r+'\Z', attr):
					if is_list: compiled += [attr]
					else: compiled[attr] = regexes[r]

		return compiled 

	def filter_not_common( self ):
		'''
		filter attributes not shared among all datasets
		(according to self.common)
		'''
		if self.verbose: 
			print("filter common assignments by whitelist:")
			print(self.common_an)
		self.out_suff += ['_filtered']
		
		studs = self.students
		removed = []
		for sn in studs:
			feats = studs[sn]
			for feat in list(feats):
				if '_' in feat: f = feat.index('_')
				if feat[0:2] == 'la' and int(feat[2:f]) not in self.common_an:
					del feats[feat]
					if feat[2:f] not in removed:
						removed += [feat[2:f]]
		if self.verbose: 
			print('{} assignments removed:'.format(len(removed)))
			print('\t'+ str(removed) )

	def filter_students_by_attr( self, regex_dict ):
		# filter students if the k matches v in regerx_dict
		if self.verbose: 
			print("filter students by attributes:")
			print(regex_dict)
		self.out_suff += ['_filtered']
		students = self.students
		attr_dict = self.compile_regexes( regex_dict ) 
		# Flag for removal
		to_be_removed = []
		for sn in students:
			for attr in attr_dict:
				if eval(str(students[sn][attr])+attr_dict[attr]):
					to_be_removed += [sn]

		for sn in to_be_removed:
			del students[sn]

	def get_curr_attr( self ):
		# return a list of the attributes held by students
		students = self.students
		sn_0 = list(students.keys())[0]
		return list(students[sn_0].keys())

	def rename_assign_chron(self):
		if self.verbose:
			print('rename assignments chronologically');

		studs = self.students
		an_cuts = self.an_cutoffs
		cuts_an = {v: k for k, v in an_cuts.items()}
		cuts = list(cuts_an)
		cuts.sort()
		an_new_name = {int(cuts_an[c]):i for i,c in enumerate(cuts)}

		for sn in studs:
			feats = studs[sn]
			for label in list(feats):
				if 'la' == label[:2]:
					an = int(label[2:label.index('_')])

					suff = label[label.index('_'):]
					new_label = "la{}{}".format(an_new_name[an], suff)
					feats[new_label] = feats[label]
					del studs[sn][label]

	def filter_attr( self , regex_list, mode='white'):
		if self.verbose: 
			print('filter attributes by {}list:'.format(mode))
			print('\t'+ str(regex_list) )

		self.out_suff += ['_filtered']
		students = self.students
		curr_attr = self.get_curr_attr() 
		attr_list = self.compile_regexes( regex_list )

		not_white = [ e for e in curr_attr if e not in attr_list ]
		for sn in students:
			if mode == 'black': filter_list = attr_list 
			elif mode ==  'white': filter_list = not_white
			for attr in filter_list:
				if attr in students[sn]:
					del students[sn][attr]

	def filter_first_n_chron( self, n, wl=True ):
		'''
		filter first n assignments chronologically
		'''
		self.out_suff += ['_filtered']
		studs = self.students
		an_cuts = self.an_cutoffs
		cuts_an = {v: k for k, v in an_cuts.items()}
		cuts = list(cuts_an)
		cuts.sort()
		filtered = [cuts_an[c] for c in cuts[n:]]
		n_f = len(filtered)
		n_k = len(an_cuts) - n_f

		# filter assignments with an in filter
		for an in filtered:
			for sn in studs:
				for label in self.la_labels:
					la = label.format(str(an))
					if la in studs[sn]:
						del studs[sn][la]

		if self.verbose: 
			ft = "whitelist" if wl else "blacklist" # filter type
			print("filter first {} assignments by {} chronologically".format(n,ft))
			print("\t"+ str(n_f) +" filtered, "+str(n_k)+" kept")

	def filter_by_date( self, cutoff ):
		self.out_suff += ['_filtered']
		students = self.students
		an_cuts = self.an_cutoffs
		filtered = []
		for an in an_cuts:
			if an_cuts[an] > cutoff: 
				filtered += [an]

		n_f = str(len(filtered))
		n_k = str(len(an_cuts)-len(filtered))

		if self.verbose: 
			print("filter by date "+str(dt.fromtimestamp( cutoff/1e3 )))
			print("\t"+n_f +" filtered, "+n_k+" kept")

		for an in filtered:
			for sn in students:
				for label in self.la_labels:
					la = label.format(str(an))
					if la in students[sn]:
						del students[sn][la]

	def classify(self, val, cond_tag, conds):
		# classify student[class] as val if conds conditions on cond_tag are met
		if (cond_tag+"_class" not in self.classes):
			self.classes[cond_tag+"_class"] = []
		self.classes[cond_tag+"_class"] += [val]

		for sn in self.students:
			conditions_met = True
			stud = self.students[sn]
			for c in conds:
				if cond_tag in stud:
					if not eval(str(stud[cond_tag])+c):
						conditions_met = False
				else: conditions_met = False
			if conditions_met: stud[cond_tag+'_class'] = val

	def classify_thresh(self,tag,sn, thresh, label):
		students = self.students
		if tag not in students[sn]: students[sn][tag] = 0
		passed = students[sn][tag] > thresh 
		if self.nominal:
			students[sn][tag+'_'+label] = True if passed else False
		else:
			students[sn][tag+'_'+label] = 1 if passed else 0

	def process(self):
		# Classify
		students = self.students
		for sn in students:
			students[sn]['is_online'] = 0
			for attr in students[sn]:
				# classify rehearse students 
				if 'la' in attr:
					try:
						an = int(attr[2:attr.index('_')])
						is_online = an in self.rehearse_an
						if is_online and int(students[sn][attr]) != self.null_int:
							students[sn]['is_online'] = 1
							break;
					except: pass

		# Build assignment labels
		if not self.la_labels:
			for attr in self.students[list(self.students.keys())[0]]:
				if 'la' in attr:
					i0 = attr.index('_')
					self.la_labels += [ 'la{}'+attr[i0:] ]

		# Track Assignment Timestamps
		for sn in self.students:
			for attr in self.students[sn]:
				if '_ts' in attr:
					f = attr.index('_')
					an = attr[2:f]
					ts = self.students[sn][attr]
					self.track_an_ts(an,ts)

		# Determine Assingment cutoffs
		for an in self.an_ts:
			non_null_ts = [ e for e in self.an_ts[an] if e != -1 ]
			self.an_cutoffs[an] = sum(non_null_ts)/len(non_null_ts)

	def show_an_cutoffs(self):
		# Sort Assginments by date

		print("-"*10)
		print(" Assignment Cutoff Dates ")
		cuts = [(k,self.an_cutoffs[k]) for k in self.an_cutoffs]
		cuts = sorted(cuts , key=lambda x: x[1])

		for cut in cuts:
			print(cut[0], dt.fromtimestamp(cut[1]/1e3))
		print("-"*10)

	def get_ans(self):
		'''
		return a list of assignment numbers
		'''

		studs = self.students
		ans = []
		for attr in list(studs[list(studs)[0]]):
			if '_' in attr: f = attr.index('_')
			if attr[0:2] == 'la' and attr[2:f] not in ans:
				ans += [attr[2:f]]
		ans.sort()
		return ans

	def load_progsnap(self):

		if self.verbose: 
			print("\tfrom progsnap")
		subdirs = get_immediate_subdirectories(self.progsnap_dir)
		for sd in subdirs[:self.n_assignments]:
			self.load_subdir(self.progsnap_dir + '/' + sd)

	def add_grades_to_student(self, row):
		# all scores are out of 1
		sid = int(row['id'])
		students = self.students

		if sid not in students: return -1

		for grade in self.totals:
			is_empty = row[grade].strip() == "";
			if is_empty: score = 0
			else: score = float(row[grade]) / float(self.totals[grade])
			students[sid][grade] = float(score)

	def load_grades(self):
		with open(self.totals_fid, 'r') as csv_file:
			totals_reader = csv.DictReader(csv_file)
			totals = list(totals_reader)[0]
			for grade in totals:
				totals[grade] = float(totals[grade])
			self.totals = totals

		with open(self.grades_fid, 'r') as csv_file:
			grades_reader = csv.DictReader(csv_file)
			for row in grades_reader:
				self.add_grades_to_student(row)

	def track_an_ts(self, an, ts):
		# keep track of the timestamps
		if an in self.an_ts: self.an_ts[an] += [ts]
		else: self.an_ts[an] = [ts]

	def track_reps(self, reps, stat): 
		if stat not in reps:
			reps[stat] = {}
			reps[stat]['max'] = float('-inf')
			reps[stat]['curr'] = 0

		if stat in reps['last']:
			reps[stat]['curr'] += 1
			if reps[stat]['curr'] > reps[stat]['max']:
				reps[stat]['max'] = reps[stat]['curr']
		else: reps[stat]['curr'] = 0


	def load_work_history(self, wh):
		num_evts = 0
		features = {}
		sn = wh.student_num()
		an = wh.assign_num()
		max_ts = 0
		min_ts = 0
		reps = {'last':None} # a dict of repeated statuses
		last = {}  # a dict of if the last event was repeated

		for evt in wh.events():
			num_evts += 1
			if evt.has("snapids"):
				tr = wh.find_testresults_event(evt.snapids()[-1])
				features['numtests'] = tr.numtests()
				features['numpassed'] = tr.numpassed()
				features['correctness'] = features['numpassed'] / features['numtests']
				if not max_ts: min_ts = tr.ts() 
				if tr.ts() < max_ts:
					raise Exception('events unordered')
				else: max_ts = tr.ts()

				for status in tr.statuses():
					if status in features:
						features[status] += 1
					else: features[status] = 0

					if reps['last']:
						self.track_reps( reps, status )

				reps['last'] = tr.statuses()

		if sn not in self.students: self.students[sn] = {}
		# stats for each lab assignment
		label_stats = { 
				'la{}_t':max_ts-min_ts,
				'la{}_s':num_evts, 
				'la{}_c':features['correctness'],
				'la{}_ts':max_ts,
				'la{}_np':features['passed'] if 'passed' in features else -1, # num passed events
				}

		status_tags = {
				'failed':'nrf', # num repeated failed
				'passed':'nrp',
				'exception':'nre'
				}

		# record the stats for the repeated statuses encountered
		for stat in status_tags:
			if stat in reps:
				m = reps[stat]['max']
				val = m if m  != float('-inf') else 0
				label_stats['la{}_'+status_tags[stat]] = val

		for label in label_stats:
			self.students[sn][label.format(str(an))] = label_stats[label]
		
	def load_assignment(self, dataset, a):
		for wh in dataset.work_histories_for_assignment(a):
			if not dataset.student_for_number(wh.student_num()).instructor():
				self.load_work_history(wh)

	def load_subdir(self, snap_path):
		dataset = progsnap.Dataset(snap_path, sortworkhistory=True)
		for a in dataset.assignments():
			if self.verbose: 
				print('loading assignment ' + a.number())
			self.load_assignment(dataset, a)

	def split_train_test(self, ratio = 0.7):
		''' 
		chooses a random selection of students to be used as train and test
		according to ratio
		where ratio*len(students) are used for training
		'''
		split = int(ratio*len(self.students))
		sids = list(self.students)
		#sids.sort()	
		random.shuffle(sids) # its better for this to be random
		self.train = sids[:split]
		self.test = sids[split:]
		if self.verbose:
			print('split train ratio: {}, {}:{}'.format(ratio, len(self.train), len(self.test)))

	def equalize_by_class(self, c):
		'''
		c: the class to equalize by
		filter out data points such that data is distributed equally by class
		'''
		fk = list(self.students[list(self.students)[0]])  # feature keys
		if c not in fk:
			raise Exception("Class: {}, is not in data".format(c))

		# SORT members for each class value
		subsets = {} 
		for sid in self.students:
			f = self.students[sid]
			if f[c] not in subsets: subsets[f[c]] = []
			subsets[f[c]] += [sid]
	
		min_count = float("inf")
		for v in subsets: 
			if len(subsets[v]) < min_count:
				min_count = len(subsets[v])
		
		equalized = {}
		for v in subsets:
			random.shuffle(subsets[v])
			for sid in subsets[v][:min_count]:
				equalized[sid] = self.students[sid]
		if self.verbose:
			print("equalized by class {}, {} kept per class:".format(c, min_count))
		n_filtered = len(self.students) - len(equalized)
		n_kept = len(equalized)
		if self.verbose:
			print("\t{} filtered, {} kept".format(n_filtered, n_kept))

		self.students = equalized

	def undersample(self, feature, value, drop_rate ):
		'''
		feature: key of a feature of students
		value: seeked value of feature
		drop_rate: float (0,1)
		drop drop_rate * len(self.students) student with f[feat] == val
		(f is the feature set for a student)
		'''
		
		studs = self.students

		total_sample = [sn for sn in studs if studs[sn][feature]==value]
		dropped = total_sample[:int(drop_rate*len(total_sample))]

		self.students = {sn:studs[sn] for sn in studs if sn not in dropped}

		if self.verbose:
			print('dropped {} samples for {}:{}, rate={}'.format(len(dropped), feature, value, drop_rate)) 

	def oversample(self, feature, value, mult ):
		'''
		feature: key of a feature of students
		value: seeked value of feature
		mult: int, number of times to replicate
		duplicate data points with feature with a val of value, with a new key
		by mult
		'''
		count = 0
		for sid in list(self.students):
			f = self.students[sid]
			if f[feature] == value:
				for i in range( mult-1 ):
					count += 1 		# count n new samples created
					self.students[int(str(sid)+str(i))] = f
		if self.verbose:
			print('oversampled {} samples for {}:{}, ({}x)'.format(count, feature, value, mult)) 
	
	def get_ranges(self):
		'''
		return a dict of ranges for all keys
		{'feature_label':[min,max]}
		'''
		ranges = {}
		for sid in self.students:
			f = self.students[sid] 
			for k in f:
				if k not in ranges: ranges[k] = [float("inf"),float("-inf")]
				# track min
				if f[k] < ranges[k][0]: ranges[k][0] = f[k]
				# track max
				if f[k] > ranges[k][1]: ranges[k][1] = f[k]

		return ranges

	def scale(self, x, r):
		'''
		x: value to be scaled
		r: [min, max] range to be scaled to
		return scaled value of x
		'''
		return float(x - r[0]) / (r[1] - r[0])

	def filter_null(self, keys = None):
		'''
		filter all students with any null data points
		keys: [list of attributes to check for null]
				check all if None
		'''
		filtered = {}

		# mark students for filtering
		for sid in self.students:
			feats = self.students[sid]
			for label in feats:
				feat = int(feats[label])
				if not keys or label in keys:
					if feat == -1:
						filtered[sid]=None

		# CREATE subset of unfiltered students		
		unfiltered = {}
		for sid in self.students:
			if sid not in filtered:
				unfiltered[sid] = self.students[sid]

		keys_info = 'All' if not keys else keys
		if self.verbose:
			print('filter students with null val for {}:'.format(keys_info))
			print('\t{} filtered, {} kept'.format(len(filtered), len(unfiltered)))

		self.students = unfiltered

	def to_xy(self,y_feat,out_path):
		'''
		y_feat = the feature to use as the class vec
		dump students data as x,y vectors
		x: features
		y: class
		'''

		data = {'x':[],'y':[]}
		train_set = {'x':[],'y':[]}
		test_set = {'x':[],'y':[]}
		# a master set of keys to maintain order
		keys = list(self.students[list(self.students)[0]].keys())
		keys.sort()

		ranges = self.get_ranges()
		n_classes = len(self.classes[y_feat])

		# split train and test if not done
		if (len(self.train)==0 and len(self.test) == 0):
			self.split_train_test()

		for sid in self.students:
			f = self.students[sid] 
			# features is all features scaled to 0,1
			feat_vec = [self.scale(f[k], ranges[k]) for k in keys if k != y_feat]
			# TO ONE HOT
			class_vec = [0] * n_classes
			class_vec[f[y_feat]] = 1

			data['x'] += [feat_vec] 
			data['y'] += [class_vec]

			if sid in self.train:
				train_set['x'] += [feat_vec] 
				train_set['y'] += [class_vec]

			if sid in self.test:
				test_set['x'] += [feat_vec] 
				test_set['y'] += [class_vec]
		
		with open(out_path+'.json','w') as f:
			json.dump(data, f)
		
		if self.arff_fid != None:
			with open(out_path+'_train.json','w') as f:
				json.dump(train_set, f)

			with open(out_path+'_test.json','w') as f:
				json.dump(test_set, f)
		
	def to_arff(self, fid=None):
		'''
		Save students data to arff file to be used by weka
		'''
		students = self.students
		sn_0 = list(students.keys())[0]
		row_headers = [h for h in students[sn_0] if h not in self.classes]
		row_headers.sort()
		is_filtered = len(self.out_suff) != 0
		headers = [] if is_filtered else ['student_num']
		headers += row_headers
		headers += self.classes # class should be last in arff

		# split train and test if not done
		if (len(self.train)==0 and len(self.test) == 0):
			self.split_train_test()

		# add null cols
		data = []
		test_set = []
		train_set = []
		file_headers = headers if is_filtered else row_headers
		for sn in students:
			row = [] if is_filtered else [sn]
			for h in file_headers:
				if h not in students[sn]:
					# convert these to ? in arff, this will fuck things up
					if is_filtered: students[sn][h] = '<NULL>' 
					else: students[sn][h] = self.null_int
				row += [students[sn][h]]
			data += [row]
			if sn in self.train: train_set += [row]
			if sn in self.test: test_set += [row]


		out_fid = "features"
		if is_filtered: out_fid = "filtered"
		if fid: out_fid = fid
	
		# write all data
		out_path = "{}/{}_{}.arff".format(self.out_dir, out_fid, self.dataset_name)
		if self.verbose and 0:
			print('Writing to {}'.format(out_path))
		arff.dump( out_path, data, relation="pcrs", names = headers)
		self.cleanse_arff( out_path, is_filtered )	
		
		if self.arff_fid != None:
			# write train
			out_path = "{}/{}_{}_train.arff".format(self.out_dir, out_fid, self.dataset_name)
			if self.verbose and 0:
				print('Writing to {}'.format(out_path))
			arff.dump( out_path, train_set, relation="pcrs", names = headers)
			self.cleanse_arff( out_path, is_filtered )	
			
			# write test
			out_path = "{}/{}_{}_test.arff".format(self.out_dir, out_fid, self.dataset_name)
			if self.verbose and 0:
				print('Writing to {}'.format(out_path))
			arff.dump( out_path, test_set, relation="pcrs", names = headers)
			self.cleanse_arff( out_path, is_filtered )	


	def cleanse_arff( self, path, is_filtered ):
		# Cleanse .arff
		f = open(path)
		lines = f.readlines()
		f.close()
		f = open(path, 'w')
		for line in lines:
			if self.nominal:
				# NOMINAL correct class attributes to nominal
				if line[0] == '@' and "{" not in line:
					parts = line.split()
					for c in self.classes:
						if c in parts:
							cs = [str(e) for e in self.classes[c]]
							parts[2] = "{"+", ".join(cs) + "}"
					line = " ".join(parts)+"\n"
	
			if is_filtered:
				#line = line.replace("-1.0","?").replace("-1","?")
				pass
			f.write(line)
		f.close()

	def load_args(self):
		if len(sys.argv) != 3:
			print('Usage: prog_to_weka <dataset_path> <output_dir>')
			exit(0)
		return sys.argv[1], sys.argv[2]


