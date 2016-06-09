'''
Convert a data set from progsnap format
supplemented with student grade data
to a weka readable format
'''

import sys, os
import arff, csv
import re
# import traceback
import subprocess
from datetime import datetime as dt

import progsnap
from helpers import *
from docopt import docopt

class DataProcessor(object):

	def __init__(self, config ):

		# Load Config
		self.fail_threshold = config['fail_threshold']
		self.n_assignments = config['n_assignments'] 
		self.grades_fid = config['grades_fid']
		self.totals_fid = config['totals_fid']
		self.dataset_path = config['progsnap_dir']
		self.out_dir = config['out_dir']
		self.arff_fid = config['arff_fid']
		
		self.students = {}
		self.an_ts = {}  # assignment number timestamps 
		self.an_cutoffs = {}
		self.la_labels = [] # TODO make consistent la -> an
		self.out_suff = []
		
		if self.arff_fid:
			self.load_arff()
		else:
			self.load_progsnap()
			self.load_grades()

		self.process()

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

	def filter_attr( self , regex_list, mode='white'):
		print('filter by {}list'.format(mode), regex_list )
		students = self.students
		sn_0 = list(students.keys())[0]
		curr_attr = list(students[sn_0].keys())

		# compile regexes
		attr_list = []
		for r in regex_list:
			for attr in curr_attr:
				if re.match(r+'\Z', attr):
					attr_list += [attr]

		not_white = [ e for e in curr_attr if e not in attr_list ]
		for sn in students:
			if mode == 'black':
				for attr in attr_list:
					if attr in students[sn]:
						del students[sn][attr]
			elif mode ==  'white':
				for attr in not_white:
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

	def process(self):
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
			non_zero_ts = [ e for e in self.an_ts[an] if e != 0 ]
			self.an_cutoffs[an] = sum(non_zero_ts)/len(non_zero_ts)

		self.show_an_cutoffs()

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
		subdirs = get_immediate_subdirectories(self.dataset_path)
		for sd in subdirs[:self.n_assignments]:
			self.load_subdir(self.dataset_path + '/' + sd)

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
		passed = students[sid]['course'] > self.fail_threshold 
		students[sid]['pass_class'] = 1 if passed else 0

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

	def load_work_history(self, wh):
		num_evts = 0
		features = {}
		sn = wh.student_num()
		an = wh.assign_num()
		max_ts = 0
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

		#features['steps'] = sum([ features[s] for s in self.status_types ])

		if sn not in self.students: self.students[sn] = {}
		
		# stats for each lab assignment
		label_stats = { 
				'la{}_s':num_evts, 
				'la{}_c':features['correctness'],
				'la{}_ts':max_ts
				}

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

	def to_arff(self):
		students = self.students
		sn = list(students.keys())[0]
		an = list(students[sn].keys())[0]
		row_headers = [h for h in students[sn]]
		row_headers.sort()
		is_filtered = len(self.out_suff) != 0
		headers = [] if is_filtered else ['student_num']
		headers += row_headers

		# add null cols
		data = []
		for sn in students:
			row = [] if is_filtered else [sn]
			for h in headers[1:]:
				if h not in students[sn]:
					# convert these to ? in arff, this will fuck things up
					#students[sn][h] = '<NULL>' 
					students[sn][h] = 0 
				row += [students[sn][h]]
			data += [row]

		out_fid = "features"
		for suff in self.out_suff:
			out_fid += suff
		if is_filtered:
			out_fid = "filtered"

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
		
if __name__ == '__main__':
	config = {
			'grades_fid':'grades.csv',
			'totals_fid':'totals.csv',
			'arff_fid':'out/features.arff',    # None to load from progsnap
			#'arff_fid':None,    # None to load from progsnap
			'out_dir':'out',
			'progsnap_dir':'progsnap_data',
			'fail_threshold':0.5,
			'n_assignments':None,  # None for all
			}

	dp = DataProcessor( config )

	#dp.filter_by_date( 1444310413490 ) # Oct 7 ->6 assngmnts
	dp.filter_by_date( 1444510413490 )
	dp.filter_attr(['a1','la.*','pass_class'])

	dp.to_arff()

