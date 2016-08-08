from data_util.processor import *

def prog_to_arff_xy(verbose=False):
	'''
	convert prog to arff and also to xy
	'''
	arff_fid ='out/features_{}.arff'.format(get_dataset_name())
	dp = DataProcessor({ 'arff_fid':arff_fid, 'verbose':verbose })

	if verbose:
		print('N Students: {}'.format(len(dp.students)))

	# FILTER out Students
	#dp.show_an_cutoffs()
	#dp.filter_students_by_attr({'is_online':'==0'})

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
		print(dp.feature_count('course_class'))
		print(dp.feature_dist('course_class'))
		print(dp.classes)

	# FILTER assignments by date
	#dp.filter_by_date( 1444310413490 ) # Oct 7 ->6 assngmnts
	#dp.filter_by_date( 1444810413490 ) # Oct 14 ->11 assngmnts

	# FILTER assignments chronologically
	dp.filter_first_n_chron( 11 ) 
	
	# RENAME assignments chronologically so that datasets allign semantically
	# DOESNT WORK
	#dp.rename_assign_chron()
	# TODO filter by asignments common to both datasets

	#dp.filter_by_date( 1454410413490 ) # Feb 2 ->8 assngmnts
	#dp.filter_by_date( 1454710413490 ) # Feb 5 ->12 assngmnts

	# FILTER nulls
	#dp.filter_null(['a1','course','exm','tst'])

	# FILTER attributes (include only these attributes)
	perform_regexes = ['.*'+str(an)+'.*' for an in dp.perform_an]
	#dp.filter_attr(['course_class']+perform_regexes)
	#dp.filter_attr(['course_class','la.*_c','la.*_s','a1'])
	#dp.filter_attr(['course_class','la.*_c','la.*_s'])
	dp.filter_attr(['course_class','la.*_c','la.*_s'])
	#dp.filter_attr(['course_class','la.*_c','la.*_s','la.*_nre','la.*_nrf'])
	#dp.filter_attr(['is_online','la.*_s','la.*_c','exm','exm_class','course_class','course'])
	#dp.filter_attr(['la.*','exm','exm_class','course_class','course'])

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

prog_to_arff_xy(verbose=True)
