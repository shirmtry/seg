from distutils.core import setup
import py2exe

setup(console=['app.py'],
      data_files=[('templates', ['templates/index.html'])],
      options={'py2exe': {'includes': ['flask', 'pydub', 'speech_recognition']}})