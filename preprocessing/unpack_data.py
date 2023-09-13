# importing the "tarfile" module
import tarfile
  
# open file
file = tarfile.open('data/CUB_200_2011.tgz')
  
# extracting file
file.extractall('./data')
  
file.close()