"""
Usage:
  bin_cfg_asts.py [--cond-op] [--flb] <dataset_path> <output_dir>

Options:
  --cond-op   Annotate control constructs with top level operator of condition
  --flb       Annotate for loops with lower-bound value
"""

import sys
import traceback
import subprocess

import cfast4py
import progsnap
from helpers import mkdirs, make_parent


if __name__ == '__main__':
    from docopt import docopt
    arguments = docopt(__doc__)
    dataset_path = arguments.get("<dataset_path>")
    output_dir = arguments.get("<output_dir>")

    dataset = progsnap.Dataset(dataset_path, sortworkhistory=True)
    print("psversion is {}".format(dataset.psversion()))
    print("there are {} assignment(s)".format(len(dataset.assignments())))
    print("there are {} student(s)".format(len(dataset.students())))

    mkdirs(output_dir)
    rpt = open("{}/report.txt".format(output_dir), 'w')

    for a in dataset.assignments():
        print("There are {} work histories in assignment {}".format(len(dataset.work_histories_for_assignment(a)), a.number()))
        for wh in dataset.work_histories_for_assignment(a):
            # Skip work histories for instructors
            if dataset.student_for_number(wh.student_num()).instructor():
                continue

            for evt in wh.events():
                if (type(evt) is progsnap.Edit) and evt.type() == 'fulltext':
                    try:
                        cfast = cfast4py.toCFAST(evt.text())
                    except:
                        if evt.has("snapids"):
                            tr = wh.find_testresults_event(evt.snapids()[0])
                            if tr.statuses()[0] == 'failed':
                                print("Parsing to CFast failed on a submission that ran without exceptions")
                        continue

                    try:
                        src_fname = "{}/code/{}_{}_{}.txt".format(output_dir, wh.assign_num(), wh.student_num(), evt.editid())
                        make_parent(src_fname)
                        with open(src_fname, 'w') as f:
                            print(evt.text(), file=f)

                        fname = "{}/cfasts/{}_{}_{}.txt".format(output_dir, wh.assign_num(), wh.student_num(), evt.editid())
                        make_parent(fname)
                        with open(fname, 'w') as f:
                            print(cfast, file=f)

                        # TODO: Uncomment if we want results stored
                        if evt.has("snapids") and evt.snapids():
                            snapid = evt.snapids()[0]
                            tr = wh.find_testresults_event(snapid)
                            if tr:
                                # Find the SHA256 hash
                                res = subprocess.getstatusoutput("sha256sum {} | cut -d ' ' -f 1".format(fname))
                                hash_output = res[1]
                                #print("hash_output={}".format(hash_output))
                                print("{}:{}:{}:{}:{}:{}:{}".format(a.number(), wh.student_num(), evt.editid(), evt.ts(), hash_output, tr.numpassed(), tr.numtests()), file=rpt)

                    except:
                        # Presumably this is just a parse error
                        # FIXME: should catch only exceptions known to be parse errors
                        traceback.print_exc(file=sys.stdout)
