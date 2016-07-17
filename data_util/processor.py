'''
Convert a data set from progsnap format
supplemented with student grade data
to a weka readable format
'''

import sys, os
import arff, csv, json
import re
# import traceback
import subprocess
from datetime import datetime as dt

import progsnap
from helpers import *
from docopt import docopt

class DataProcessor(object):

	def __init__(self, config={} ):

		# Load Config
		self.load_config( config )

		# assigment numbers
		# TODO move this to an external file
		self.prepare_an = []
		self.rehearse_an = [87,131,43,88,92]
		self.perform_an = [82,55,49,54,48,41,47,35,36,37,45,39,59,64,56,62,42,61,63,57,58,91,67]
		
		self.students = {}
		self.an_ts = {}  # assignment number timestamps 
		self.an_cutoffs = {}
		self.la_labels = [] # TODO make consistent la -> an
		self.out_suff = []

		self.batch_ptr = 0
		self.null_int = -1

		if self.arff_fid:
			self.load_arff()
		else:
			self.load_progsnap()
			self.load_grades()

		self.process()

	def load_config(self, config ):
		'''
		LOAD data config
		use defaults as needed
		'''
		# TODO ... is this a good place to put this?
		defaults = {
            'grades_fid':'data/grades.csv',
            'totals_fid':'data/totals.csv',
            'arff_fid':'out/features.arff',    # None to load from progsnap
            'out_dir':'out',
            'progsnap_dir':'data/progsnap_data',
            'n_assignments':None,  # None for all
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

	def load_arff(self):
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

	def filter_students_by_attr( self, regex_dict ):
		# filter students if the k matches v in regerx_dict
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


	def filter_attr( self , regex_list, mode='white'):
		print('filter attributes by {}list:'.format(mode))
		print( regex_list )

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
					an = attr[2:4]
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


	def load_progsnap(self):
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
		reps = {'last':None} # a dict of repeated statuses
		last = {}  # a dict of if the last event was repeated

		for evt in wh.events():
			num_evts += 1
			if evt.has("snapids"):
				tr = wh.find_testresults_event(evt.snapids()[-1])
				features['numtests'] = tr.numtests()
				features['numpassed'] = tr.numpassed()
				features['correctness'] = features['numpassed'] / features['numtests']
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
			print('loading assignment ' + a.number())
			self.load_assignment(dataset, a)

	def to_xy(self,y_feat,out_path):
		'''
		y_feat = the feature to use as the class vec
		Return students data as x,y vectors
		x: features
		y: class
		'''
		x = []
		y = []
		for sid in self.students:
			f = self.students[sid]
			feat_vec = [f[k] for k in f if k != y_feat]
			class_vec = [1,0] if f[y_feat] else [0,1]
			x += [feat_vec] 
			y += [class_vec]

		with open(out_path,'w') as f:
			json.dump({'x':x,'y':y}, f)
		
	def to_arff(self, fid=None):
		'''
		Save students data to arff file to be used by weka
		'''
		students = self.students
		sn_0 = list(students.keys())[0]
		row_headers = [h for h in students[sn_0]]
		row_headers.sort()
		is_filtered = len(self.out_suff) != 0
		headers = [] if is_filtered else ['student_num']
		headers += row_headers

		# add null cols
		data = []
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

		out_fid = "features"
		if is_filtered: out_fid = "filtered"
		if fid: out_fid = fid

		out_path = "{}/{}.arff".format(self.out_dir, out_fid)
		arff.dump( out_path, data, relation="pcrs", names = headers)

		# Cleanse .arff
		f = open(out_path)
		lines = f.readlines()
		f.close()
		f = open(out_path, 'w')
		for line in lines:
			f.write(line.replace("'<NULL>'","?"))
		f.close()

	def load_args(self):
		if len(sys.argv) != 3:
			print('Usage: prog_to_weka <dataset_path> <output_dir>')
			exit(0)
		return sys.argv[1], sys.argv[2]


