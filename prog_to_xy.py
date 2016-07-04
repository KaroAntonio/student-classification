from data_processor import DataProcessor
from data_processor import get_config

#LOAD
dp = DataProcessor( get_config() )
dp.filter_attr(['course_class','la.*_c','la.*_s'])
dp.to_xy('course_class','out/xy.json')
