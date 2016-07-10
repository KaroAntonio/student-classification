from data_util.processor import DataProcessor
from data_util.config import get_config

dp = DataProcessor( get_config() )

# Filter out Students
#dp.show_an_cutoffs()
#dp.filter_students_by_attr({'is_online':'==0'})

# classify course_class as 2 if student['course'] >=0.8
'''
dp.classify(2,'course',['>0.8'])
dp.classify(1,'course',['>=0.5','<=0.8'])
dp.classify(0,'course',['<=0.5'])
'''
dp.classify(1,'course',['>=0.5'])
dp.classify(0,'course',['<0.5'])
# Filter out attributes
#dp.filter_by_date( 1444310413490 ) # Oct 7 ->6 assngmnts
dp.filter_by_date( 1444810413490 ) # Oct 14 ->11 assngmnts
#perform_regexes = ['.*'+str(an)+'.*' for an in config['perform_an']]
#dp.filter_attr(['course_class']+perform_regexes)
dp.filter_attr(['course_class','la.*_c','la.*_s'])
#dp.filter_attr(['is_online','la.*_s','la.*_c','exm','exm_class','course_class','course'])
#dp.filter_attr(['la.*','exm','exm_class','course_class','course'])

dp.to_arff()
