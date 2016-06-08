import io
import functools
import csv
import json
import os
import errno
import sys

from pycparser import c_parser, c_ast, plyparser

from collections import OrderedDict


########## Base Submission Class ##########
class Submission():
    def __init__(self, student_id, status, code):
        self.student_id = student_id
        self.status = status
        self.code = code.strip()

    def __cleanup_code(self):
        '''pycparser requires comments and includes to be removed.
        '''
        preprocess_c = []
        start_stop = 0
        in_comment = False
        for line in self.code.split("\n"):
            line = line.strip()

            if in_comment:
                # Assuming comments end a line.
                if "*/" in line:
                    in_comment = False
                    line = line[line.find("*/") + 2:]
                else:
                    continue

            if line.startswith("//"):
                # PCRS specific -- it omits starter code
                if "Implementation start" in line:
                    line = "int petersen_start{0};".format(start_stop)
                    start_stop += 1
                elif "Implementation end" in line:
                    line = "int petersen_end{0};".format(start_stop)
                    start_stop += 1
                else:
                    continue

            # Doesn't handle multiple multi-line comments in a line.
            if "/*" in line:
                in_comment = True
                start = line[:line.find("/*")]
                end = ''

                if "*/" in line:
                    in_comment = False
                    end = line[line.find("*/") + 2:]

                line = start + end

            if not line.startswith("#"):
               line = line.replace("\\", "\\\\")
               preprocess_c.append(line)
        return "\n".join(preprocess_c)

    def ast(self):
        if self.status == "CompileError":
            raise RuntimeError("Code does not compile")

        cleaned_code = self.__cleanup_code()

        try:
            parser = c_parser.CParser()
            ast = parser.parse(cleaned_code)
            return ast
        except plyparser.ParseError as e:
            raise RuntimeError("Plyparse error: {0}".format(e))

    def ast_str(self):
        a = self.ast()

        # Standardizing variable names
        anonymizer = NameWalker()
        anonymizer.visit(a)

        str_buf = io.StringIO()
        a.show(buf=str_buf)

        return str_buf.getvalue()

    def __str__(self):
        return "{0} submitted ({1}):\n{2}".format(self.student_id, self.status, self.code)


########## Base AST Counter Class ##########
@functools.total_ordering
class ASTCounter():
    def __init__(self, ast):
        self.ast = ast

        self.total = 0
        self.passed = 0
        self.student_ids = set()
        self.submits = []

    def add_submit(self, submit):
        self.total += 1
        self.passed += int(submit.status.lower() == "pass")
        self.student_ids.add(submit.student_id)
        self.submits.append(submit)

    # Not a great order function, but we just want the count
    def __eq__(self, other):
        return self.total == other.total and self.ast == other.ast

    def __lt__(self, other):
        return self.total < other.total or (self.total == other.total and self.ast < other.ast)

    def __str__(self):
        return "{0} submitters ({1} passed, {2} failed)\n{3}".format(len(self.student_ids), self.passed, self.total - self.passed, self.ast)


########## Base Category Class ##########
@functools.total_ordering
class Category():
    def __init__(self, category_id):
        self.category_id = category_id

        self.total = 0
        self.passed = 0
        self.student_ids = set()
        self.asts = []

    def add_ast(self, ast):
        self.total += ast.total
        self.passed += ast.passed
        self.student_ids.update(ast.student_ids)
        self.asts.append(ast)

    def __eq__(self, other):
        return self.total == other.total and self.category_id == other.category_id

    def __lt__(self, other):
        return self.total < other.total or (self.total == other.total and self.category_id < other.category_id)

    def __str__(self):
        ast = None if len(self.asts) == 0 else self.asts[0].ast
        return "Category {0} -- {1} submits ({2} passed) comprising {3} asts and {4} students\n{5}"\
               .format(self.category_id, self.total, self.passed, len(self.asts), len(self.student_ids), ast)


########## pycparser helpers ##########
class NameWalker(c_ast.NodeVisitor):
    def __init__(self, normalize_names=True):
        self.names = {}
        self.uniq = 0
        self.normalize_names = normalize_names

    def __add_name__(self, name, replace_name):
        if "petersen" in name:                  # Hack to keep markers in place for PCRS exercises
            self.names[name] = name
        elif name not in self.names:
            if replace_name:
                self.names[name] = "var{0}".format(self.uniq)
                self.uniq += 1
            else:
                self.names[name] = name

    def visit_FuncDef(self, node):
        func_name = node.decl.name
        self.__add_name__(func_name, False)

        oldnames = self.names
        self.names = self.names.copy()

        self.visit(node.decl)
        if node.body.block_items:
            for c in node.body.block_items:
                self.visit(c)

        self.names = oldnames

    def visit_Decl(self, node):
        self.__add_name__(node.name, self.normalize_names)
        node.name = self.names[node.name]

        for c in node.children():
            self.visit(c[1])

    def visit_PtrDecl(self, node):
        self.visit(node.type)

    def visit_TypeDecl(self, node):
        node.declname = self.names.get(node.declname, node.declname)

    def visit_ID(self, node):
        node.name = self.names.get(node.name, node.name)


class CFASTNode:
    def __init__(self, kind, identifier=''):
        self.kind = kind
        self.children = []
        self.identifier = identifier
        # other info?

    # Recursively convert to a JSON object (really a dict)
    def to_json_obj(self):
        node = OrderedDict()
        node['kind'] = self.kind
        if self.identifier:
            node['identifier'] = self.identifier
        if self.children:
            kids = []
            for kid in self.children:
                kids.append(kid.to_json_obj())
            node['children'] = kids
        return node

    # Convert to a JSON string representation.
    def to_json_str(self):
        return json.dumps(self.to_json_obj(), indent=2)

    # Dump tree representation to given stream
    def dump(self, out=sys.stdout):
        self._dump_tree(out, 0)

    def add_annotation(self, val):
        if self.identifier:
            self.identifier += ','
        self.identifier += str(val)

    def _dump_tree(self, out, indent):
        print("  " * indent, end="", file=out)
        print(self.kind, end="", file=out)
        if self.identifier:
            print("[{0}]".format(self.identifier), end="", file=out)
        print("", file=out)
        for child in self.children:
            child._dump_tree(out, indent + 1)


# A universal set object.  (Well, you can't actually enumerate its members.)
class UniversalSet:
    def __init__(self):
        pass

    def __contains__(self, member):
        return True

    def __str__(self):
        return "All"


# Visitor to construct a CFAST from an AST.
# All nodes of the CFAST are instances of CFASTNode.
# TODO: provide some configurability regarding what information
# is retained.
class CFASTWalker(c_ast.NodeVisitor):
    # Control flow node types.
    # Note that Compound is necessary to enable recursive
    # visitation of nested code (i.e., loop bodies, etc.)
    CONTROL_FLOW = ['For', 'If', 'While', 'DoWhile', 'Break', 'Continue', 'Return', 'Compound']

    # Constructs that have a condition
    HAS_CONDITION = frozenset(['For', 'If', 'While', 'DoWhile'])

    def __init__(self):
        self._show_cond_op = False
        self._show_for_loop_lower_bound = False

    def show_cond_op(self):
        self._show_cond_op = True

    def show_for_loop_lower_bound(self):
        self._show_for_loop_lower_bound = True

    # Generic visit method: creates a CFASTNode and adds
    # CFASTs for all children recursively.
    def generic_visit(self, node):
        return self._do_visit(node)

    # Accumulate children by recursively converting children of
    # given AST node into CFASTs and adding them to given CFAST
    # node.  The "retain" keyword argument can be used to
    # specify which types of children should be retained.
    def _do_visit(self, node, **kwargs):
        # Which child nodes should be retained: None means "all"
        retain = frozenset(kwargs['retain']) if ('retain' in kwargs) else UniversalSet()
        cfast_node = CFASTNode(node.__class__.__name__)
        for cname, child in node.children():
            should_retain = child.__class__.__name__ in retain
            #print("{0}: retain={1}, should_retain={2}".format(child.__class__.__name__, retain, should_retain))
            if should_retain:
                cfast_node.children.append(self.visit(child))

        # If this is a for loop, and we're showing for loop lower bounds,
        # attempt to extract the lower bound and annotate the CFAST node
        # with it.
        if self._show_for_loop_lower_bound and type(node) is c_ast.For:
            if type(node.init) is c_ast.Assignment:
                rv = node.init.rvalue
                if type(rv) is c_ast.Constant:
                    cfast_node.add_annotation(rv.value)
            elif type(node.init) is c_ast.DeclList:
                first_decl = node.init.decls[0]
                if first_decl.init:
                    rv = first_decl.init
                    if type(rv) is c_ast.Constant:
                        cfast_node.add_annotation(rv.value)

        # If this is a loop or if statement, and we're showing
        # the top level operator of conditions, annotate the CFAST
        # node with the operator.
        if self._show_cond_op and node.__class__.__name__ in CFASTWalker.HAS_CONDITION:
            if type(node.cond) is c_ast.BinaryOp:
                #cfast_node.identifier = str(node.cond.op)
                cfast_node.add_annotation(node.cond.op)

        return cfast_node

    def visit_FileAST(self, node):
        return self._do_visit(node, retain=['FuncDef'])

    def visit_FuncDef(self, node):
        result = self._do_visit(node, retain=['Compound'])
        (cname, child) = node.children()[0]
        result.identifier = child.name # decorate the FuncDef node with the function name
        return result

    def visit_If(self, node):
        return self._visit_control(node)

    def visit_While(self, node):
        return self._visit_control(node)

    def visit_For(self, node):
        return self._visit_control(node)

    def visit_DoWhile(self, node):
        return self._visit_control(node)

    def visit_Compound(self, node):
        return self._visit_control(node)

    def _visit_control(self, node):
        # Retain just control-flow statements.
        # TODO: allow other stuff to be retained
        return self._do_visit(node, retain=CFASTWalker.CONTROL_FLOW)

    def visit_Return(self, node):
        # We could retain info about the returned value here, but for now we don't.
        return self._do_visit(node, retain=[])


# NodeVisitor whose job is to normalize ASTs to make it more
# likely that equivalent control-flow will result in identical
# CFASTs.  Right now we just insert a Compound anywhere that
# one is missing: iftrue and iffalse in If, stmt in For,
# While, and DoWhile.
class NormalizeControlFlow(c_ast.NodeVisitor):
    def __init__(self, retain=CFASTWalker.CONTROL_FLOW):
        self._retain = frozenset(retain)

    def visit_If(self, node):
        node.iftrue = self._fix(node.iftrue)
        if node.iffalse:
            node.iffalse = self._fix(node.iffalse)

        # recursively continue visitation
        self.visit(node.iftrue)
        if node.iffalse:
            self.visit(node.iffalse)

    def visit_DoWhile(self, node):
        self._visit_loop(node)

    def visit_For(self, node):
        self._visit_loop(node)

    def visit_While(self, node):
        self._visit_loop(node)

    def _visit_loop(self, node):
        node.stmt = self._fix(node.stmt)

        # recursively continue visitation
        self.visit(node.stmt)

    def _fix(self, node):
        if type(node) is c_ast.Compound and len(node.block_items) == 1 and self._should_retain(node.block_items[0]):
            # We have a Compound with a single child which is a control construct.
            # In this case, we can simplify the AST by removing the Compound
            # and just using the control construct.  This is in keeping
            # with the desire to make the CFASTs as simple as possible.
            return node.block_items[0]

        elif type(node) is c_ast.Compound or self._should_retain(node):
            # We have a nested Compound or control construct:
            # just leave it as-is.
            return node

        else:
            # This is a node that would not normally be retained
            # (e.g., a bare statement), so create a Compound to ensure
            # that it isn't lost.
            compound = c_ast.Compound([node], node.coord)
            return compound

    def _should_retain(self, node):
        return node.__class__.__name__ in self._retain


# Splitting an empty string on a delimiter yields a list
# with a single empty string, which is not a useful behavior
# in our case.
def _split(s, sep):
    result = s.split(sep)
    if result == ['']:
        result = []
    return result


# Information about tree edit distance (relative to the previous submission.)
# As generated by cfastTed.jar and appearing in report_ted.txt.
class TreeEditDistance:
    def __init__(self, info):
        self._info = info
        subfields = info.split('|')
        self.distance = float(subfields[0])
        self.deletions = _split(subfields[1], ',')
        self.insertions = _split(subfields[2], ',')

        # Replacements are pairs [orig, curr] indicating that
        # a CFAST node with label orig was changed to a node
        # with the label curr.
        self.replacements = list(map(lambda spec: spec.split(';'), _split(subfields[3], ',')))

    def __str__(self):
        return self._info


# Entry in a "report.txt" file in an analysis directory.
# Each entry represents information about a compilable submission.
# Stringifies as a line of text in the same format
# as the original report line.
class ReportEntry:
    # Initialize from a line of the report file:
    # either as a string with fields delimited by ':',
    # or as a list of fields
    def __init__(self, line):
        if type(line) is str:
            fields = line.split(':')
            self._init_from_fields(fields)
        elif type(line) is list:
            self._init_from_fields(line)
        else:
            raise RuntimeError("Can't initialize ReportEntry from {}".format(line.__class__.__name__))

    def _init_from_fields(self, fields):
        self.assign_num = int(fields[0])
        self.student_num = int(fields[1])
        self.editid = int(fields[2])
        self.ts = float(fields[3])
        self.cfast_hash = fields[4]
        self.numpassed = int(fields[5])
        self.numtests = int(fields[6])
        # Add tree edit distance information if present
        self.ted = TreeEditDistance(fields[7]) if len(fields) > 7 else None

    def __str__(self):
        result = "{}:{}:{}:{}:{}:{}:{}".format(self.assign_num, self.student_num, self.editid, self.ts, self.cfast_hash, self.numpassed, self.numtests)
        if self.ted:
            result += ":{}".format(self.ted)
        return result

    def is_correct(self):
        return self.numtests > 0 and self.numpassed == self.numtests

    def is_partially_correct(self):
        return self.numpassed > 0


# History is a wrapper for a list of ReportEntry objects.
# It has some useful methods for, e.g., finding the first
# correct submission in a history.
class History:
    def __init__(self, entries):
        self._entries = entries

    def entries():
        return self._entries

    # Find first entry matching given predicate.
    # If no entries match, returns None.
    def find_first(self, pred):
        for entry in self._entries:
            if pred(entry):
                return entry
        return None

    # Return the first correct entry, or None if there are
    # no correct entries
    def first_correct(self):
        return self.find_first(lambda e: e.is_correct())

    # Return the first entry with specified CFAST hash.
    def first_with_cfast(self, cfast_hash):
        return self.find_first(lambda e: e.cfast_hash == cfast_hash)

    # Find the percentage of specified entry in the history's
    # sequence of steps (entries).  If the history has a single
    # entry, the returned percentage is 100.
    def step_pct(self, entry):
        def findit(x):
            res = x is entry
            #print("Check {} is {}: {}".format(id(x), id(entry), res))
            return res

        where = find_first(self._entries, findit)
        #print("where={}".format(where))
        if where < 0:
            raise RuntimeError("Could not find entry in History!")
        return (where/(len(self._entries)-1))*100.0 if len(self._entries) > 1 else 100.0


# A Session is a sequence of report entries falling within
# the same "work session".
class Session:
    def __init__(self, begin_ts, end_ts, history):
        self._begin_ts = begin_ts
        self._end_ts = end_ts
        self._history = history

    # Beginning of session (as timestamp.)
    def begin_ts(self):
        return self._begin_ts

    # End of session (as timestamp.)
    def end_ts(self):
        return self._end_ts

    # Does this session include the given timestamp?
    def includes_ts(self, ts):
        return ts >= self.begin_ts() and ts <= self.end_ts()

    # Duration of session in milliseconds.
    def duration(self):
        return self.end_ts() - self.begin_ts()


# Class representing the entire report.txt.
# Consists of a sequence of ReportEntry objects.
# Has support for providing the submission histories for each
# student.
class Report:
    def __init__(self, f):
        self._entries = []
        for line in f:
            line = line.rstrip()
            self._entries.append(ReportEntry(line))

    # For iterating over entries
    def __iter__(self):
        return self._entries.__iter__()

    # For iterating over histories: relies on the report
    # storing submission contiguously by student.
    def histories(self):
        prev_entry = None
        history = []

        for cur_entry in self._entries:
            if prev_entry and prev_entry.student_num != cur_entry.student_num:
                # Current entry is for a different student
                # than the previous entry, so yield the
                # current history, then start a new one.
                yield history
                prev_entry = None
                history = []

            history.append(cur_entry)
            prev_entry = cur_entry

        if history:
            yield history

    def _entries_between(self, history, begin_ts, end_ts):
        entries = []
        for entry in history:
            if entry.ts >= begin_ts and entry.ts <= end_ts:
                entries.append(entry)
        return entries

    # Convert a history (list of report entries) into a list
    # of Sessions.  Note that the progsnap Dataset is needed
    # in order to analyze fine-grained edit events to determine
    # where each session begins and ends.  Any events that
    # are separated by more than maxpause milliseconds are
    # assumed to be in different sessions.
    def work_sessions_for_history(self, history, dataset, maxpause):
        # If necessary convert History object back to list of entries
        if history is History:
            history = history.entries()

        student = dataset.student_for_number(history[0].student_num)
        assignment = dataset.assignment_for_number(history[0].assign_num)
        wh = dataset.work_history_for_student_and_assignment(student, assignment)

        sessions = []

        begin_ts = None
        end_ts = None

        for evt in wh.events():
            # See if there is a session underway

            if begin_ts:
                # There is a session underway: see if it continues or ends
                if evt.ts() - begin_ts > maxpause:
                    # Session ends: grab the ReportEntries, create Session,
                    # start a new session
                    session_entries = self._entries_between(history, begin_ts, end_ts)
                    sessions.append(Session(begin_ts, end_ts, session_entries))
                    begin_ts = evt.ts()
                else:
                    # Session continues
                    pass
            else:
                # No session underway, so start one
                begin_ts = evt.ts()

            # Mark (currently known) end of current session
            end_ts = evt.ts()

        # Finish whatever session is currently underway.
        if begin_ts:
            session_entries = self._entries_between(history, begin_ts, end_ts)
            sessions.append(Session(begin_ts, end_ts, session_entries))

        return sessions

    # Find the time (in milliseconds) of given report entry
    # within the student's work history as indicated by the
    # list of sessions.
    def time_in_work_history(self, entry, sessions):
        time_in_history = 0

        for session in sessions:
            # See if the specified report entry falls within this session
            if session.includes_ts(entry.ts):
                # Return accumulated time in history plus offset in session
                return time_in_history + (entry.ts - session.begin_ts())
            time_in_history += session.duration()

        raise RuntimeError("Could not find work session for entry {}".format(entry.editid))


# An entry in summary.txt
class SummaryEntry:
    def __init__(self, line):
        if type(line) is str:
            fields = line.split(':')
            self._init_from_fields(fields)
        elif type(line) is list:
            self._init_from_fields(line)
        else:
            raise RuntimeError("Can't initialize SummaryEntry from {}".format(line.__class__.__name__))

    def _init_from_fields(self, fields):
        self.cfast_hash = fields[0]
        self.numpartial = int(fields[1])
        self.numcorrect = int(fields[2])
        self.total = int(fields[3])

    def __str__(self):
        return "{}:{}:{}:{}".format(self.cfast_hash[:8], self.numpartial, self.numcorrect, self.total)

    def __repr__(self):
        return self.__str__()

    # What fraction of submissions in this CFAST were correct?
    # (0=none, 1=all)
    def correctness(self):
        return self.numcorrect / self.total

    # Were any of the submissions in this CFAST correct?
    def any_correct(self):
        return self.numcorrect > 0


# Data loaded from summary.txt
class Summary:
    def __init__(self, f):
        self._entries = []

        for line in f:
            self._entries.append(SummaryEntry(line))

    def __iter__(self):
        return self._entries.__iter__()

    def entries(self):
        return self._entries

    # Get the entry with the largest percentage of correct submissions
    def most_likely_correct(self):
        ranked = sorted(self._entries, key=lambda entry: entry.correctness(), reverse=True)
        return ranked[0]

    # Get the entry with the largest absolute number of correct submissions
    def most_correct(self):
        ranked = sorted(self._entries, key=lambda entry: entry.numcorrect, reverse=True)
        return ranked[0]

    # Get the top X% of CFASTs by observed submissions.
    def most_likely(self, top_pct):
        ranked = sorted(self._entries, key=lambda entry: entry.total, reverse=True)
        #print("ranked={}".format(ranked))
        total = len(ranked)
        # For example: if top_pct is 20, then we take the top 20 percent of entries
        num_to_take = int(total * (top_pct/100))
        #print("Taking {} cfasts".format(num_to_take))
        return ranked[:num_to_take]


######################################################################
# Common utility functions
######################################################################


# This file reader function assumes the data is a csv in the format:
# submission_id,user_id,problem_id,timestamp,code,status,compile_messages,test_result
def read_file(fname):
    students = set()
    problems = {}    # problem_id: [list of submissions]
    with open(fname) as code_f:
        code_f.readline()     # Removing header
        code_r = csv.reader(code_f, delimiter=',', quotechar='"')
        for submit in code_r:
            problems.setdefault(submit[2], []).append(Submission(submit[1], submit[5], submit[4]))
            students.add(submit[1])
    print("{0} problems read from {1} students".format(len(problems), len(students)))
    return problems


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


# Make given directory and parent directories if they do
# not already exist.
def mkdirs(path):
    # See: http://stackoverflow.com/questions/18973418
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise exc
        pass


# Ensure that parent directory of given path exists.
def make_parent(path):
    slash = path.rfind('/')
    if slash > 0:
        parent = path[:slash]
        mkdirs(parent)


# Pare down a CFAST by removing the FileAST and FuncDef
# nodes.  This is possible only for CFASTs that contain
# a single function.
def pare_down_cfast(cfast):
    n = cfast
    if n.kind == 'FileAST' and len(n.children) == 1:
        n = n.children[0].children[0]
    return n


# Find index of first list element matching predicate.
# Returns -1 if no element matches the predicate.
# Seems like there should be a built-in for this.
def find_first(lst, pred):
    index = 0
    for elt in lst:
        if pred(elt):
            return index
        index += 1
    return -1


# Put a value in a histogram array.
# Specified histogram bin increases by specified delta (default is 1.)
def put_in_bin(val, bins, binsize, delta=1):
    binend = 0
    numbins = len(bins)
    for i in range(numbins):
        binend += binsize
        if val <= binend:
            bins[i] += delta
            return
    raise RuntimeError("No bin found for value {}".format(val))


# vim:set expandtab:
# vim:set tabstop=4:
