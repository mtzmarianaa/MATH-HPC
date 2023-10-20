import numpy

# Build a binary tree structure

class Node:
   def __init__(self, data):
      self.left = None
      self.right = None
      self.data = data

   def insert(self, data):
# Compare the new value with the parent node
      if self.data:
         if data < self.data:
            if self.left is None:
               self.left = Node(data)
            else:
               self.left.insert(data)
         elif data > self.data:
               if self.right is None:
                  self.right = Node(data)
               else:
                  self.right.insert(data)
      else:
         self.data = data

# Print the tree
   def PrintTree(self):
      if self.left:
         self.left.PrintTree()
      print( self.data),
      if self.right:
         self.right.PrintTree()


def tsqr(A, blocksize):
    '''
    Serial implementation of the TSQR algorithm
    '''
    array = []
    m = A.shape[0]
    i = 0
    numBlocks = m / blocksize
    while (i < numBlocks):
        array.append(A[(i * blocksize):((i + 1) * blocksize), :])
        i = i + 1
    i = 1
    temp = numpy.empty((2 * blocksize, A.shape[1]))
    temp[0:blocksize, :] = array[0]
    temp[blocksize:(2 * blocksize), :] = array[1]
    theR = numpy.linalg.qr(temp, 'r')
    # print "MADE IT THIS FAR" 
    n = A.shape[1]
    while (i < numBlocks - 1):
        temp = numpy.empty((blocksize + theR.shape[0], n))
        temp[0:theR.shape[0]] = theR
        temp[theR.shape[0]:(theR.shape[0]+blocksize)] = array[i + 1]
        theR = numpy.linalg.qr(temp, 'r')
        i = i + 1
    return theR
