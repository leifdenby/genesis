import imp
import os
import glob

def package_contents():
    for path in glob.glob(os.path.join(os.path.dirname(__file__), '*.py')):
        path = os.path.basename(path)
        if not path.startswith('_'):
            yield __package__ + '.' + path.replace('.py', '')


print("Available modules:")
print()

for module in package_contents():
    print("\t" + module)
