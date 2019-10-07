from setuptools import setup, find_packages

setup(name='gym-push',
      packages=find_packages(),
      package_data={
        '': ['*.csv', '*.npy'],
      },      
      version='0.0.9',
      install_requires=['gym', 'numpy', 'pandas', 'joblib', 'eel', 'json-tricks'],  # And any other dependencies foo needs,
      author='Kieran Fraser',
      author_email='kfraser@tcd.ie',
) 