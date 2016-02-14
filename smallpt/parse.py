import os, sys

filename = "output.txt"
#newline = os.pathsep
newline = '\n'

argc = len(sys.argv)
print(argc)
print(sys.argv)

if argc > 1:
	filename = sys.argv[1]
	
print(filename)

parsed = open(filename + ".csv", "w+")

def insertNewline():
	global parsed
	parsed.write('\n')

def value(line):
	return line[line.find(':') + 1:].strip()
	
def integer(line):
	return str(int(value(line)))
	
def floating(line):
	return str(float(value(line))).replace('.', ',')
	
def insert(val):
	global parsed
	val = str(val)
	parsed.write("'" + val + "'" + ',')
	
tests = {}
name = ""
samples = 0
allSamples = {}

def trimName(name):
	getRidOf = ["T1", "CPU", "(R)", "(TM)", "@"]
	for s in getRidOf:
		name = name.replace(s + " ", " ")
	name = name.replace("org", "Original")
	name = name.replace("openmp", "Original OpenMP")
	name = name.replace("_single", "Single Precision")
	for i in range(0, 3):
		name = name.replace("  ", " ")
	return name

def dumpTests():
	global tests
	global name
	global samples
	global allSamples
	if len(tests) > 0:
		names = tests.keys()
		samples = list(allSamples.keys())
		samples.sort()
		insert("")
		for s in samples:
			insert(s)
		insertNewline()
		for name in names:
			insert(name)
			for s in samples:
				if s in tests[name].keys():
					insert(tests[name][s])
				else:
					insert("")
			insertNewline()
	name = ""
	samples = 0
	tests = {}
	allSamples = {}

with open(filename) as file:
	for line in file:
		try:
			if line.startswith("smallpt SYCL tester"):
				dumpTests()
				insertNewline()
			elif line.startswith("samples per pixel:"):
				samples = int(integer(line))
				allSamples[samples] = 0
			elif line.startswith("Running test:"):
				name = trimName(value(line))
				if name not in tests.keys():
					tests[name] = {}
			elif line.startswith("time:"):
				tests[name][samples] = floating(line)
		except Exception as e:
			print("Exception: {0}".format(e))

dumpTests()
parsed.close()
