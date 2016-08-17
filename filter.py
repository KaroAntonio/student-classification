from data_util.processor import *

def show_common():
	# FIND COMMON ASSIGNMENTS
	arff_fid ='out/features_{}.arff'.format('201601')
	dp = DataProcessor({ 'arff_fid':arff_fid, 'verbose':verbose })
	feats1 = dp.students[list(dp.students)[0]]

	arff_fid ='out/features_{}.arff'.format('201509')
	dp = DataProcessor({ 'arff_fid':arff_fid, 'verbose':verbose })
	feats2 = dp.students[list(dp.students)[0]]

	common = [feat for feat in feats1 if feat in feats2]
	common = [feat for feat in feats2 if feat in common]
	common = set([ int(e[2:4]) for e in common if e not in ['course', 'is_online']])
	common = list(common)
	common.sort()
	print(common)

def to_arff_xy(dataset_name=None, verbose=False):
	'''
	convert prog to arff and also to xy
	'''

	# LOAD
	if not dataset_name: dataset_name = get_dataset_name()
	print('---DATASET: {}---'.format(dataset_name))

	arff_fid ='out/features_{}.arff'.format(dataset_name)

	dp = DataProcessor({
		'dataset_name':dataset_name, 
		'arff_fid':arff_fid, 
		#'nominal':False,
		'verbose':verbose 
		})

	# FILTER nulls
	#dp.filter_null(['a1','course','exm','tst'])
	#dp.filter_null(['course'])

	# CLASSIFY
	# classify course_class as 2 if student['course'] >=0.8
	'''
	dp.classify(2,'course',['>=0.8'])
	dp.classify(1,'course',['>=0.5','<0.8'])
	dp.classify(0,'course',['<0.5'])
	'''
	dp.classify(1,'course',['>=0.5'])
	dp.classify(0,'course',['<0.5'])
	if verbose:
		print('N Students: {}'.format(len(dp.students)))
		#dp.disp_feat_stats('la.*_t')
		#dp.disp_feat_stats('la.*_s')
		print('n / class',dp.feature_count('course_class'))
		print('ratio / class',dp.feature_dist('course_class'))
		print('classes', dp.classes)

	# FILTER out Students
	#dp.show_an_cutoffs()
	#dp.filter_students_by_attr({'is_online':'==0'})

	#dp.filter_not_common()

	# FILTER assignments by date
	#dp.filter_by_date( 1444310413490 ) # Oct 7 ->6 assngmnts
	#dp.filter_by_date( 1444810413490 ) # Oct 14 ->11 assngmnts

	#dp.filter_by_date( 1454410413490 ) # Feb 2 ->8 assngmnts
	#dp.filter_by_date( 1454710413490 ) # Feb 5 ->12 assngmnts

	# FILTER assignments chronologically
	#dp.filter_first_n_chron( 11 ) 
	
	# FILTER attributes (include only these attributes)
	perform_regexes = ['.*'+str(an)+'.*' for an in dp.perform_an]
	#dp.filter_attr(['course_class']+perform_regexes)
	#dp.filter_attr(['course_class','la.*_c','la.*_s','a1'])
	dp.filter_attr(['course_class','la.*_c','la.*_s'])
	#dp.filter_attr(['course_class','la.*_c','la.*_s','la.*_t'])
	#dp.filter_attr(['course_class','la.*_c','la.*_s','la.*_nre','la.*_nrf'])
	#dp.filter_attr(['is_online','la.*_s','la.*_c','exm','exm_class','course_class','course'])
	#dp.filter_attr(['la.*','exm','exm_class','course_class','course'])
	if verbose:
		print('n features: {}'.format(dp.n_features()))

	# EQUALIZE
	dp.equalize_by_class('course_class')

	# OVERSAMPLE to bias the data
	#dp.oversample('course_class',1,2)
	#dp.oversample('course_class',0,3)

	# UNDERSAMPLE
	#dp.undersample('course_class',1,1./3.)
	
	dp.split_train_test(0.7)

	dp.to_arff()
	dp.to_xy('course_class','out/xy_{}'.format(dp.dataset_name))

#to_arff_xy('201601', verbose=True)
#to_arff_xy('201509', verbose=True)
to_arff_xy('s2014', verbose=True)
to_arff_xy('k2015', verbose=True)
