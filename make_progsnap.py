#!/usr/bin/env python3
"""
Usage:
  make_progsnap.py <dataset_name> <codedata_filename>
"""

######################################################################
# Progsnap metadata constants
_metadata = [
    ('contact', 'Andrew Petersen'),
    ('email', 'andrew.petersen@utoronto.ca'),
    ('psversion', '0.1-dev'),
    ('courseurl', 'https://student.utm.utoronto.ca/calendar/course_detail.pl?Depart=7&Course=CSC108H5'),
]


######################################################################
# Includes

import os
import json
from collections import namedtuple, OrderedDict
import csv


######################################################################
# Reading datafiles
Submission = namedtuple('Submission', 'submit_id user_id problem_id timestamp submission status result')

def read_codedata(code_fname):
    problems = []
    num_failed = 0
    with open(code_fname) as code_f:
        code_f.readline()
        code_r = csv.reader(code_f, delimiter=',', quotechar='"')
        for entry in code_r:
            try:
                problems.append(Submission(*entry))
            except:
                num_failed += 1
    print("Number of entries failed to read {}.".format(num_failed))
    return problems


######################################################################
# Progsnap utilities

def make_progsnap(problem_id, problems, students):
    _make_progsnap_dirs(problem_id)
    os.chdir(problem_id)    # all later files should be in the problem directory

    _make_progsnap_metadata(problem_id)
    _make_progsnap_students(students)
    _make_progsnap_history(problem_id, problems)

    os.chdir('..')          # returning to original cwd

class TagEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Tag):
            return obj.fields
        return super().default(obj)

class Tag():
    def __init__(self, tag, value):
        self.fields = OrderedDict()
        self.fields['tag'] = tag
        self.fields['value'] = value

    def __str__(self):
        return '{0}\n'.format(json.dumps(self, cls=TagEncoder, separators=(',',':')))

def _make_progsnap_dirs(name):
    os.mkdir(name)
    os.mkdir('{0}{1}assignment'.format(name, os.sep))
    os.mkdir('{0}{1}history'.format(name, os.sep))
    os.mkdir('{0}{1}history{1}{0:0>4}'.format(name, os.sep))

# TODO: a bit of a hack, since each exercise is in its own snapshot. If an entire
# course were put into a progsnap, then this metadata creation should occur higher.
def _make_progsnap_metadata(name):
    metadata = _metadata[:]
    metadata.append(('name', '{0} (problem #{1})'.format(metadata.pop()[1], name)))
    with open('dataset.txt', 'w') as outf:
        for (tag, val) in metadata:
            outf.write(str(Tag(tag, val)))

    with open('assignments.txt', 'w') as outf:
        assign = OrderedDict()
        assign['number'] = name
        assign['path'] = 'assignment{0}{1:0>4}.txt'.format(os.sep, name)
        outf.write(str(Tag('assignment', assign)))

    with open('assignment{0}{1:0>4}.txt'.format(os.sep, name), 'w') as outf:
        assign = OrderedDict()
        assign['name'] = 'Exercise #{0}'.format(name)

        # TODO: language name is a stub -- obtain from database
        assign['language'] = 'python'
        assign['url'] = 'https://teach.cdf.toronto.edu/StG108/problems/python/{0}/submit'.format(name)

        # TODO: dates omitted. Obtain from database
        assign['assigned'] = 0
        assign['due'] = 0

        # TODO: testcases omitted entirely

        for (key, value) in assign.items():
            outf.write(str(Tag(key, value)))

def _make_progsnap_students(students):
    with open('students.txt', 'w') as outf:
        for student in students:
            stu_info = OrderedDict()
            stu_info['number'] = int(student)
            stu_info['instructor'] = False
            outf.write(str(Tag('student', stu_info)))

def _make_progsnap_history(problem_id, problems):
    for problem in problems:
        if problem.problem_id == problem_id:
            with open('history{0}{1:0>4}{0}{2:0>4}.txt'.format(os.sep, problem_id, problem.user_id), 'a') as outf:
                # First, the edit event
                sub_info = OrderedDict()
                # TODO: convert timestamp to seconds
                sub_info['ts'] = problem.timestamp
                sub_info['editid'] = int(problem.submit_id)
                sub_info['filename'] = 'answer.py'
                # TODO: relax fulltext restriction
                sub_info['type'] = 'fulltext'
                sub_info['text'] = problem.submission
                sub_info['snapids'] = [sub_info['editid']]
                outf.write(str(Tag('edit', sub_info)))

                # And now the submission event
                sub_info = OrderedDict()
                # TODO: convert timestamp to seconds
                sub_info['ts'] = problem.timestamp
                sub_info['snapid'] = int(problem.submit_id)
                outf.write(str(Tag('submission', sub_info)))

                # And the testresults event
                # TODO: actually encode test data. I've folded them all into a single test for now.
                sub_info['numtests'] = 1
                sub_info['numpassed'] = int(problem.status == 'Pass')
                sub_info['result'] = problem.result
                if problem.status == 'Pass':
                    status = 'passed'
                else:
                    if 'Error' in problem.result.split()[0]:
                        status = 'exception'
                    elif 'Timeout' == problem.result.split()[0]:
                        status = 'timeout'
                    else:
                        status = 'failed'
                sub_info['statuses'] = [status]
                outf.write(str(Tag('testresults', sub_info)))


######################################################################
# Produce students file


if __name__ == '__main__':
    import docopt
    arguments = docopt.docopt(__doc__)
    _metadata.append(('name', arguments.get('<dataset_name>')))
    problems = read_codedata(arguments.get('<codedata_filename>'))

    # TODO: obtain student data from a combination of marks file and submissions
    student_ids = set([problem.user_id for problem in problems])

    problem_ids = set([problem.problem_id for problem in problems])
    for problem_id in problem_ids:
        make_progsnap(problem_id, problems, student_ids)
