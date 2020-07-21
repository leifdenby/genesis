import imp
import os
import glob

def package_contents():
    paths = glob.glob(os.path.join(os.path.dirname(__file__), '*/__init__.py'))
    for path in paths:
        path = os.path.basename(path.replace('/__init__.py', ''))
        if not path.startswith('__main'):
            yield __package__ + '.' + path


print("Available modules:")
print()

for module in package_contents():
    print("\t" + module)
