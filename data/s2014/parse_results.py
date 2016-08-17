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
		course_grade = parts[headers.index('final-grade')+1].split(',')
	except:
		print(parts[headers.index('final-grade')+1])
		course_grade = ['','']
	try:
		if len(course_grade) == 2:
			stud = [sn, course_grade[0],course_grade[1]]
		else:
			if course_grade[0] == '#N/A':
				course_grade[0] = ''
			stud = [sn, course_grade[0],'']

	except:
		print(course_grade)
		break
	grades_data += [stud]	

tsv_file.close()

grades_file = open(out_fid,'w')

grades_file.write(','.join(['id','course','course_2']) + '\n')
for stud in grades_data:
	grades_file.write(','.join(stud) + '\n')

grades_file.close()
			

