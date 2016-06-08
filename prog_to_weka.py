'''
Convert a data set from progsnap format
supplemented with student grade data
to a weka readable format
'''

import sys, os, csv
import arff
import traceback
import subprocess

import progsnap
from helpers import *
from docopt import docopt

class DataProcessor(object):

	def __init__(self, config ):

		# Load Config
		self.status_types = config['status_types']
		self.totals = config['totals']
		self.fail_threshold = config['fail_threshold']
		self.n_assignments = config['n_assignments'] 
		self.grades_file = config['grades_file']
		
		# Load Args
		self.dataset_path, out_dir = self.load_args()

		self.students = {}

		self.load_progsnap()
		self.load_grades()

		# Write Students to weka
		self.to_arff(out_dir)

	def load_progsnap(self):
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
			students[sid][grade] = score
		passed = students[sid]['course'] > self.fail_threshold 
		students[sid]['pass_class'] = 1 if passed else 0

	def load_grades(self):
		with open(self.grades_file, 'r') as csv_file:
			grades_reader = csv.DictReader(csv_file)
			for row in grades_reader:
				self.add_grades_to_student(row)

	def process_work_history(self, wh):
		num_evts = 0
		features = {}
		sn = wh.student_num()
		an = wh.assign_num()
		for s in self.status_types:
			features[s] = 0
		for evt in wh.events():
			num_evts += 1
			if evt.has("snapids"):
				tr = wh.find_testresults_event(evt.snapids()[-1])
				features['numtests'] = tr.numtests()
				features['numpassed'] = tr.numpassed()
				features['correctness'] = features['numpassed'] / features['numtests']
				for status in tr.statuses():
					features[status] += 1
			features['steps'] = sum([ features[s] for s in self.status_types ])
		
		if sn not in self.students: self.students[sn] = {}
		
		# stats for each lab assignment
		self.students[sn]['la'+str(an)+'_s'] = features['steps']
		self.students[sn]['la'+str(an)+'_c'] = features['correctness']
		self.students[sn]['la'+str(an)+'_ne'] = num_evts 
		
	def process_assignment(self, dataset, a):
		for wh in dataset.work_histories_for_assignment(a):
			if not dataset.student_for_number(wh.student_num()).instructor():
				self.process_work_history(wh)

	def load_subdir(self, snap_path):
		dataset = progsnap.Dataset(snap_path, sortworkhistory=True)
		for a in dataset.assignments():
			print('loading assignment ' + a.number())
			self.process_assignment(dataset, a)

	def to_arff(self, out_dir):

		students = self.students
		sn = list(students.keys())[0]
		an = list(students[sn].keys())[0]
		row_headers = [h for h in students[sn]]
		row_headers.sort()
		headers = ['student_num']
		headers += row_headers

		# add null cols
		data = []
		for sn in students:
			row = [sn]
			for h in headers[1:]:
				if h not in students[sn]:
					# convert these to ? in arff, this will fuck things up
					students[sn][h] = 0
				row += [students[sn][h]]
			data += [row]

		arff.dump("{}/features.arff".format(out_dir), data, relation="features", names = headers)

	def load_args(self):
		if len(sys.argv) != 3:
			print('Usage: prog_to_weka <dataset_path> <output_dir>')
			exit(0)
		return sys.argv[1], sys.argv[2]
		
if __name__ == '__main__':
	config = {
			'grades_file':'grades.csv',
			'fail_threshold':0.5,
			'n_assignments':None,  # None for all
			'status_types':['passed','failed','exception','timeout'],
			'totals':{ 
				'a1':46, 
				'a2':52, 
				'a3':20, 
				'tst': 24, 
				'exm':75, 
				'course':100 
				}, # what each grade is out of
			}

	dp = DataProcessor( config );


