import os, pickle, sys

datadir = '__data__'

def fetch( datasetName, func ):
   '''try to fetch from local filename, use func to get it if not'''
   filename = datasetName + '.pickle'
   path = os.path.join( datadir, filename )

   if not os.path.exists( datadir ):
      os.mkdir( datadir )
   
   if os.path.exists( path ):
      f = open( path, 'rb' )
      try:
         data = pickle.load( f )
         f.close()
         return data
      except:
         f.close()
      
   f = open( path, 'wb' )
   data = func()
   pickle.dump( data, f, protocol=pickle.HIGHEST_PROTOCOL )
   f.close()
   return data