''' 
script to convert U Helsinki grade results tsv into UoT style grades.csv 
'''

in_fid = 'k2015-results.tsv'
out_fid = 'grades.csv'

tsv_file = open(in_fid, 'r')

line = tsv_file.readline()

# next two lines are nothing
tsv_file.readline()
tsv_file.readline()

headers = line.strip().split('\t')
headers[0] = 'student'
grades_data = []
for line in tsv_file:
	parts = line.strip().split('\t')
	sn = parts[0]		
	try:
		course_grade = parts[headers.index('pass-hyv-or-fail-hyl')].strip()
	except:
		course_grade = ''
	if course_grade == 'HYL':
		course_grade = '30'
	elif course_grade == 'HYV':
		course_grade == '100'
	
	stud = [sn, course_grade]

	grades_data += [stud]	

tsv_file.close()

grades_file = open(out_fid,'w')

grades_file.write(','.join(['id','course']) + '\n')
for stud in grades_data:
	grades_file.write(','.join(stud) + '\n')

grades_file.close()
			

