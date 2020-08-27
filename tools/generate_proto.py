from distutils import spawn

import glob
import os
import posixpath
import shutil
import subprocess
import sys

class GeneratePyProtos():
  """Generate MediaPipe Python protobuf files by Protocol Compiler."""

  def run(self):
    if 'PROTOC' in os.environ and os.path.exists(os.environ['PROTOC']):
      self._protoc = os.environ['PROTOC']
    else:
      self._protoc = spawn.find_executable('protoc')
    if self._protoc is None:
      sys.stderr.write(
          'protoc is not found. Please run \'apt install -y protobuf'
          '-compiler\' (linux) or \'brew install protobuf\'(macos) to install '
          'protobuf compiler binary.')
      sys.exit(-1)
    # Build framework protos.
    for proto_file in glob.glob(
        'mediapipe/framework/**/*.proto', recursive=True):
      if proto_file.endswith('test.proto'):
        continue
      proto_dir = os.path.dirname(os.path.abspath(proto_file))
      if proto_dir.endswith('testdata'):
        continue
      init_py = os.path.join(proto_dir, '__init__.py')
      if not os.path.exists(init_py):
        sys.stderr.write('adding necessary __init__ file: %s\n' % init_py)
        open(init_py, 'w').close()
      self._generate_proto(proto_file)

  def _generate_proto(self, source):
    """Invokes the Protocol Compiler to generate a _pb2.py."""

    output = source.replace('.proto', '_pb2.py')
    sys.stderr.write('generating proto file: %s\n' % output)
    if (not os.path.exists(output) or
        (os.path.exists(source) and
         os.path.getmtime(source) > os.path.getmtime(output))):

      if not os.path.exists(source):
        sys.stderr.write('cannot find required file: %s\n' % source)
        sys.exit(-1)

      protoc_command = [self._protoc, '-I.', '--python_out=.', source]
      if subprocess.call(protoc_command) != 0:
        sys.exit(-1)

GeneratePyProtos().run()