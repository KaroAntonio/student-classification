from data_processor import DataProcessor
from data_processor import get_config

#LOAD
dp = DataProcessor( get_config() )
dp.filter_by_date( 1444310413490 ) # Oct 7 ->6 assngmnts
#dp.filter_by_date( 1443321413490 ) # Oct 7 ->6 assngmnts
dp.filter_attr(['exm_class','la.*_c','la.*_s'])
dp.to_xy('exm_class','out/xy.json')
