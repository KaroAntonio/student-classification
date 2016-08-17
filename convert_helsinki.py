'''
script to clean the *-out.tsv file for U Helsinki
combine it with the grade data
save it to a csv with allllll the infos
'''

dataset = 'k2015'

# paths assume we run from root directory
in_fid = 'data/{}/{}-out.tsv'.format(dataset,dataset)
grades_fid = 'data/{}/grades.csv'.format(dataset)

out_fid = 'data/{}/features.csv'.format(dataset)

grades = {}
grades_file = open(grades_fid,'r')

# LOAD GRADES
grades_file.readline() #skip header
for line in grades_file:
	parts = line.strip().split(',')
	sn = parts[0] if len(parts[0]) == 8 else parts[0][1:]
	grades[sn] = parts[1:]
grades_file.close()

def assign_grade(stud):
	mg = 75.0 # assumed
	sn = str(stud['id'])
	if sn in grades:
		g = grades[sn]
		c0 = float(g[0]) / mg if g[0].strip() != '' else -1
		studs[sn]['course'] = c0
		if len(grades[sn]) == 2:
			c1 = float(g[1]) / mg if g[1].strip() != '' else -1
			studs[sn]['course_2'] = c1 
	else: 
		studs[sn]['course'] = -1
		if len(grades.values()[0])==2:
			studs[sn]['course_2'] = -1

f = open(in_fid, 'r')
n_lines = 0
studs = {}
exs_ids = {}
for line in f:
	n_lines += 1
	parts = line.strip().split('\t')
	sn = parts[2][1:]
	compiles = parts[-2]=='true'
	if parts[-1]=='null':
		c_ = 0
	else:
		c_ = float(parts[-1]) # new correctness
	ts = int(parts[0])
	ex = parts[3]
	
	if ex not in exs_ids: 
		exs_ids[ex] = len(exs_ids)

	ex_id = exs_ids[ex]
	s_tag = 'la{}_s'.format(ex_id)
	c_tag = 'la{}_c'.format(ex_id)
	ts_tag = 'la{}_ts'.format(ex_id)
	mints_tag = 'la{}_mints'.format(ex_id)

	if sn not in studs: studs[sn] = {
			'id':sn
			}

	if 'course' not in studs[sn]:
		assign_grade(studs[sn])

	if s_tag not in studs[sn]: studs[sn][s_tag] = 0
	if c_tag not in studs[sn]: studs[sn][c_tag] = 0
	if ts_tag not in studs[sn]: studs[sn][ts_tag] = float('-inf')
	if mints_tag not in studs[sn]: studs[sn][mints_tag] = float('inf')

	studs[sn][s_tag] += 1
	if c_ > studs[sn][c_tag]: studs[sn][c_tag] = c_
	if ts > studs[sn][ts_tag]: studs[sn][ts_tag] = ts
	if ts < studs[sn][mints_tag]: studs[sn][mints_tag] = ts
	
	'''	
	if n_lines >= 100000:
		break
	'''
	if n_lines % 5000 == 0:
		print( len(studs) , n_lines)

f.close()

# FILL IN NULLS
all_feats = {}
for sn in studs:
	for feat in studs[sn]:
		if feat not in all_feats:
			all_feats[feat] = 0

n_null = 0
n_non_null = 0
for sn in studs:
	for feat in all_feats:
		if feat not in studs[sn]:
			n_null += 1
			studs[sn][feat] = -1
		else:
			n_non_null+=1

print('n_null: {}'.format(n_null))
print('n_non_null: {}'.format(n_non_null))

s = len(studs[sn])

for sn in studs:
	if len(studs[sn]) != s:
		print('WRONG N FEATS', len(studs[sn]),  s)
		exit(0)

out_file = open(out_fid, 'w')
headers = list(studs[sn])
out_file.write(','.join(headers)+'\n')
for sn in studs:
	line = ','.join([ str(studs[sn][h]) for h in headers])
	out_file.write(line+'\n')
out_file.close()




